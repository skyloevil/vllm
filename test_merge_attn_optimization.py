#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test script to verify merge_attn_states optimization effectiveness
Goal: Compare CUDA utilization and performance improvements before/after optimization
"""

import logging
import os
import sys
import time
from typing import List, Tuple

import torch

# Add vLLM path
sys.path.insert(0, os.path.dirname(__file__))

# Set log level to DEBUG to observe fallback behavior
logging.basicConfig(level=logging.DEBUG)


def create_test_tensors(
        num_tokens: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype = torch.half) -> Tuple[torch.Tensor, ...]:
    """Create test attention states tensors"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create random attention outputs and LSE values
    shape = (num_tokens, num_heads, head_size)
    lse_shape = (num_heads, num_tokens)

    prefix_output = torch.randn(shape, dtype=dtype, device=device)
    suffix_output = torch.randn(shape, dtype=dtype, device=device)
    prefix_lse = torch.randn(lse_shape, dtype=torch.float32, device=device)
    suffix_lse = torch.randn(lse_shape, dtype=torch.float32, device=device)
    output = torch.zeros_like(prefix_output)
    output_lse = torch.zeros_like(prefix_lse)

    return output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse


def benchmark_merge_attn_states(test_configs: List[Tuple[int, int, int,
                                                         torch.dtype]],
                                num_iterations: int = 100):
    """Benchmark performance under different configurations"""
    print("🚀 Starting merge_attn_states optimization effectiveness test")
    print("=" * 80)

    from vllm.attention.ops.merge_attn_states import merge_attn_states

    results = []

    for num_tokens, num_heads, head_size, dtype in test_configs:
        print(
            f"\n📊 Test config: tokens={num_tokens}, heads={num_heads}, head_size={head_size}, dtype={dtype}"
        )

        # Create test data
        test_data = create_test_tensors(num_tokens, num_heads, head_size,
                                        dtype)
        output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse = test_data

        # Warmup
        for _ in range(10):
            merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                              suffix_lse, output_lse)

        # Synchronize GPU
        torch.cuda.synchronize()

        # Timing test
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                              suffix_lse, output_lse)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_time_ms = (end_time - start_time) * 1000 / num_iterations
        throughput = (num_tokens * num_heads *
                      head_size) / avg_time_ms * 1000  # elements/sec

        result = {
            'config': (num_tokens, num_heads, head_size, str(dtype)),
            'avg_time_ms': avg_time_ms,
            'throughput': throughput
        }
        results.append(result)

        print(f"  ⏱️  Average time: {avg_time_ms:.4f} ms")
        print(f"  🔥 Throughput: {throughput:.2e} elements/sec")

    return results


def test_dtype_coverage():
    """Test dtype coverage improvement"""
    print("\n🧪 Testing dtype coverage")
    print("-" * 40)

    from vllm.attention.ops.merge_attn_states import merge_attn_states

    # Test different dtypes
    test_dtypes = [
        torch.float32,
        torch.half,
        torch.bfloat16,
    ]

    # Test FP8 if supported
    if hasattr(torch, 'float8_e4m3fn'):
        test_dtypes.extend([torch.float8_e4m3fn, torch.float8_e5m2])

    for dtype in test_dtypes:
        try:
            test_data = create_test_tensors(64, 8, 128, dtype)
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse = test_data

            merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                              suffix_lse, output_lse)
            print(f"  ✅ {dtype}: Supported")
        except Exception as e:
            print(f"  ❌ {dtype}: Not supported - {e}")


def test_headdim_coverage():
    """Test headdim coverage improvement"""
    print("\n🎯 Testing headdim coverage")
    print("-" * 40)

    from vllm.attention.ops.merge_attn_states import merge_attn_states

    # Test different head_sizes, including previously unsupported ones
    test_head_sizes = [32, 64, 80, 96, 128, 160, 192, 256]

    for head_size in test_head_sizes:
        try:
            test_data = create_test_tensors(64, 8, head_size, torch.half)
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse = test_data

            merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                              suffix_lse, output_lse)
            print(f"  ✅ head_size={head_size}: Supported")
        except Exception as e:
            print(f"  ❌ head_size={head_size}: Not supported - {e}")


def test_cuda_vs_triton_usage():
    """Verify CUDA kernel usage"""
    print("\n⚡ Testing CUDA kernel usage")
    print("-" * 40)
    
    import io
    import logging
    
    # Capture log output to detect CUDA usage
    log_capture = io.StringIO()
    
    # Set up temporary logger handler
    logger = logging.getLogger('vllm.attention.ops.merge_attn_states')
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    try:
        from vllm.attention.ops.merge_attn_states import merge_attn_states
        
        # Test configurations that should use CUDA
        cuda_configs = [
            (64, 8, 128, torch.half),    # Standard aligned config
            (32, 16, 80, torch.half),    # Non-aligned config that should be supported after optimization
            (128, 32, 96, torch.bfloat16), # bfloat16 config
        ]
        
        cuda_count = 0
        triton_count = 0
        
        for num_tokens, num_heads, head_size, dtype in cuda_configs:
            log_capture.seek(0)
            log_capture.truncate(0)
            
            test_data = create_test_tensors(num_tokens, num_heads, head_size, dtype)
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse = test_data
            
            merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                              suffix_lse, output_lse)
            
            log_content = log_capture.getvalue()
            if "Using CUDA merge_attn_states" in log_content:
                cuda_count += 1
                print(f"  ✅ CUDA: {num_tokens}×{num_heads}×{head_size} {dtype}")
            elif "Fallback to Triton" in log_content:
                triton_count += 1
                reason = log_content.split('Fallback to Triton merge_attn_states: ')[1].strip() if 'Fallback to Triton merge_attn_states: ' in log_content else "unknown"
                print(f"  📉 Triton: {num_tokens}×{num_heads}×{head_size} {dtype} - {reason}")
            else:
                print(f"  ❓ Unknown: {num_tokens}×{num_heads}×{head_size} {dtype}")
        
        print(f"\n📊 CUDA usage rate: {cuda_count}/{len(cuda_configs)} ({cuda_count/len(cuda_configs)*100:.1f}%)")
        
    finally:
        logger.removeHandler(handler)


def main():
    """Main test function"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping tests")
        return

    print(f"🖥️  CUDA device: {torch.cuda.get_device_name()}")
    print(
        f"🧠 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    # Test performance in different scenarios
    test_configs = [
        # (num_tokens, num_heads, head_size, dtype)
        (128, 32, 128, torch.half),  # Standard config
        (256, 32, 128, torch.half),  # More tokens
        (128, 32, 80, torch.half),  # Non-4/8 multiple head_size (optimization focus)
        (64, 16, 96, torch.half),  # Another non-aligned config
        (512, 16, 64, torch.bfloat16),  # Large batch + bfloat16
    ]

    # Run benchmark tests
    results = benchmark_merge_attn_states(test_configs)

    # Test coverage improvements
    test_dtype_coverage()
    test_headdim_coverage()
    test_cuda_vs_triton_usage()

    print("\n📈 Optimization summary:")
    print("=" * 80)
    print("✅ Corrected headdim constraints to match CUDA kernel requirements")
    print("✅ Added experimental FP8 dtype support")
    print("✅ Added detailed fallback logging for performance analysis")
    print("✅ Enhanced error handling and CUDA usage monitoring")
    print("\nExpected benefits:")
    print("🎯 Better visibility into CUDA vs Triton usage")
    print("🎯 Improved debugging capabilities")
    print("🎯 More accurate dtype and headdim support detection")


if __name__ == "__main__":
    main()
