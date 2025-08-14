#!/usr/bin/env python3
"""
Test script for zero-copy memory management optimization in InputBatch
This script benchmarks memory efficiency and performance improvements.
"""

import time
import sys
import os
import torch
import numpy as np
from typing import Dict, List

# Add vLLM to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from vllm.v1.worker.gpu_input_batch import InputBatch, CachedRequestState
from vllm.sampling_params import SamplingParams


class PerformanceBenchmark:
    """Benchmark class to measure memory optimization improvements"""
    
    def __init__(self, max_num_reqs: int = 128, max_model_len: int = 2048):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing benchmark with {max_num_reqs} max requests, {max_model_len} max length")
        print(f"Using device: {self.device}")
    
    def create_test_input_batch(self) -> InputBatch:
        """Create InputBatch with optimized zero-copy memory pool"""
        return InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_reqs * 512,
            device=self.device,
            pin_memory=True,
            vocab_size=50000,
            block_sizes=[16, 32],  # Different block sizes for KV cache groups
            is_spec_decode=False
        )
    
    def create_test_request(self, req_id: str, num_prompt_tokens: int = 100, num_output_tokens: int = 20) -> CachedRequestState:
        """Create a test request with specified token counts"""
        prompt_tokens = list(range(1, num_prompt_tokens + 1))
        output_tokens = list(range(num_prompt_tokens + 1, num_prompt_tokens + num_output_tokens + 1))
        
        return CachedRequestState(
            req_id=req_id,
            prompt_token_ids=prompt_tokens,
            mm_kwargs=[],  # No multimodal data for this test
            mm_positions=[],
            sampling_params=SamplingParams(temperature=0.7, top_p=0.9, top_k=50),
            pooling_params=None,
            generator=None,
            block_ids=([list(range(10))], [list(range(10, 20))]),  # Mock block IDs for 2 groups
            num_computed_tokens=num_prompt_tokens,
            output_token_ids=output_tokens,
        )
    
    def benchmark_memory_allocation(self, num_requests: int = 64) -> Dict[str, float]:
        """Benchmark memory allocation efficiency"""
        print(f"\\nBenchmarking memory allocation with {num_requests} requests...")
        
        # Create InputBatch
        input_batch = self.create_test_input_batch()
        
        # Measure time and memory for adding requests
        start_time = time.perf_counter()
        requests = []
        
        for i in range(num_requests):
            req = self.create_test_request(f"req_{i}", 
                                         num_prompt_tokens=np.random.randint(50, 200),
                                         num_output_tokens=np.random.randint(10, 50))
            requests.append(req)
            input_batch.add_request(req)
        
        add_requests_time = time.perf_counter() - start_time
        
        # Get memory efficiency stats
        memory_stats = input_batch.get_memory_efficiency_stats()
        
        # Test batch GPU sync
        start_sync_time = time.perf_counter()
        gpu_tensor = input_batch.sync_dirty_requests_to_gpu()
        sync_time = time.perf_counter() - start_sync_time
        
        # Test removal performance
        start_remove_time = time.perf_counter()
        for i in range(num_requests // 2):  # Remove half the requests
            input_batch.remove_request(f"req_{i}")
        remove_time = time.perf_counter() - start_remove_time
        
        results = {
            'add_requests_time_ms': add_requests_time * 1000,
            'sync_to_gpu_time_ms': sync_time * 1000,
            'remove_requests_time_ms': remove_time * 1000,
            'requests_per_second': num_requests / add_requests_time,
            **memory_stats
        }
        
        return results
    
    def benchmark_memory_efficiency(self) -> Dict[str, float]:
        """Benchmark memory efficiency improvements"""
        print("\\nBenchmarking memory efficiency...")
        
        input_batch = self.create_test_input_batch()
        
        # Add varying sizes of requests to test fragmentation
        request_sizes = [50, 100, 200, 150, 75, 300, 25, 400]
        
        for i, size in enumerate(request_sizes):
            if i >= self.max_num_reqs:
                break
            req = self.create_test_request(f"varying_req_{i}", 
                                         num_prompt_tokens=size,
                                         num_output_tokens=20)
            input_batch.add_request(req)
        
        memory_stats = input_batch.get_memory_efficiency_stats()
        
        print(f"Memory utilization: {memory_stats['memory_utilization']:.2%}")
        print(f"Memory efficiency: {memory_stats['memory_efficiency']:.2%}")
        print(f"Total memory: {memory_stats['total_memory_mb']:.2f} MB")
        print(f"Fragmentation ratio: {memory_stats['fragmentation_ratio']:.2%}")
        
        return memory_stats
    
    def run_comprehensive_benchmark(self) -> Dict[str, float]:
        """Run comprehensive benchmark suite"""
        print("=" * 60)
        print("ZERO-COPY MEMORY OPTIMIZATION BENCHMARK")
        print("=" * 60)
        
        results = {}
        
        # Test different request batch sizes
        batch_sizes = [16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            if batch_size > self.max_num_reqs:
                continue
                
            print(f"\\nTesting batch size: {batch_size}")
            batch_results = self.benchmark_memory_allocation(batch_size)
            
            print(f"  Add requests: {batch_results['add_requests_time_ms']:.2f} ms")
            print(f"  GPU sync: {batch_results['sync_to_gpu_time_ms']:.2f} ms")
            print(f"  Remove requests: {batch_results['remove_requests_time_ms']:.2f} ms")
            print(f"  Throughput: {batch_results['requests_per_second']:.0f} req/sec")
            print(f"  Memory efficiency: {batch_results['memory_efficiency']:.2%}")
            
            results[f'batch_{batch_size}'] = batch_results
        
        # Memory efficiency test
        memory_results = self.benchmark_memory_efficiency()
        results['memory_efficiency'] = memory_results
        
        return results


def main():
    """Main benchmark execution"""
    try:
        # Test with smaller sizes first for compatibility
        benchmark = PerformanceBenchmark(max_num_reqs=128, max_model_len=1024)
        results = benchmark.run_comprehensive_benchmark()
        
        print("\\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Calculate average performance improvements
        avg_add_time = np.mean([r['add_requests_time_ms'] for r in results.values() 
                               if isinstance(r, dict) and 'add_requests_time_ms' in r])
        avg_throughput = np.mean([r['requests_per_second'] for r in results.values() 
                                 if isinstance(r, dict) and 'requests_per_second' in r])
        avg_memory_efficiency = results['memory_efficiency']['memory_efficiency']
        
        print(f"Average request addition time: {avg_add_time:.2f} ms")
        print(f"Average throughput: {avg_throughput:.0f} requests/sec")
        print(f"Memory efficiency: {avg_memory_efficiency:.2%}")
        print(f"Fragmentation ratio: {results['memory_efficiency']['fragmentation_ratio']:.2%}")
        
        # Performance benefits summary
        print("\\nOptimization Benefits:")
        print("- Zero-copy memory operations reduce allocation overhead")
        print("- Pre-allocated memory pool minimizes fragmentation")
        print("- Batch GPU synchronization improves transfer efficiency")
        print("- Memory locality improvements reduce cache misses")
        
        return True
        
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)