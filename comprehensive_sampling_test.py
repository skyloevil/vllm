#!/usr/bin/env python3
"""
Comprehensive test script for vLLM v1 sampling optimization.
Tests both correctness and performance of the optimized sampling implementation.
"""
import torch
import time
import sys
from typing import Dict, Optional


def random_sample_original(
    probs: torch.Tensor,
    generators: Dict[int, torch.Generator],
) -> torch.Tensor:
    """Original implementation for comparison."""
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # Original slow approach
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def random_sample_optimized(
    probs: torch.Tensor,
    generators: Dict[int, torch.Generator],
) -> torch.Tensor:
    """Optimized implementation with dynamic threshold."""
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # Optimized batch processing for requests with custom generators
        generator_indices = list(generators.keys())
        generator_objects = list(generators.values())
        
        # Dynamic threshold: higher for larger batches where vectorization pays off more
        batch_size = probs.shape[0]
        vectorization_threshold = min(4 + batch_size // 32, 8)
        
        if len(generator_indices) > vectorization_threshold:
            # Vectorized approach for large batches
            generator_tensor = torch.stack([
                torch.empty_like(q[i]).exponential_(generator=gen)
                for i, gen in zip(generator_indices, generator_objects)
            ])
            q[generator_indices] = generator_tensor
        else:
            # Original approach for small batches to avoid overhead
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def simulate_flashinfer_sample(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    generators: Dict[int, torch.Generator],
) -> torch.Tensor:
    """Mock FlashInfer sample for testing."""
    # Simplified implementation for testing
    assert not generators, "FlashInfer doesn't support generators"
    
    if k is not None:
        # Apply top-k filter
        top_k_values, top_k_indices = torch.topk(logits, k.max().item(), dim=-1)
        mask = torch.arange(logits.size(-1), device=logits.device).expand(logits.size(0), -1)
        mask = mask >= k.unsqueeze(1)
        logits = logits.masked_fill(mask, float('-inf'))
    
    if p is not None:
        # Apply top-p filter (simplified)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum_probs > p.unsqueeze(1)
        mask[:, 0] = False  # Keep at least one token
    
    # Sample from the remaining distribution
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def test_correctness():
    """Test correctness of optimizations."""
    print("="*60)
    print("CORRECTNESS TESTS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    # Test 1: FlashInfer usage correctness
    print("\n1. Testing FlashInfer usage scenarios:")
    
    batch_size = 8
    vocab_size = 1000
    logits = torch.randn(batch_size, vocab_size, device=device)
    k_values = torch.randint(1, 100, (batch_size,), device=device)
    p_values = torch.rand(batch_size, device=device) * 0.5 + 0.5
    
    # Test no generators (should work)
    try:
        result = simulate_flashinfer_sample(logits.clone(), k_values, p_values, {})
        print(f"   ✅ FlashInfer without generators: {result.shape}")
    except Exception as e:
        print(f"   ❌ FlashInfer failed: {e}")
    
    # Test with generators (should fail safely)
    generators = {0: torch.Generator(device=device), 2: torch.Generator(device=device)}
    try:
        result = simulate_flashinfer_sample(logits.clone(), k_values, p_values, generators)
        print(f"   ❌ Should have failed with generators")
    except AssertionError:
        print(f"   ✅ Correctly rejected FlashInfer with generators")
    
    # Test 2: Sampling function correctness
    print("\n2. Testing sampling function correctness:")
    
    test_cases = [
        (16, 4),    # Small batch, few generators
        (32, 10),   # Medium batch, medium generators  
        (64, 20),   # Large batch, many generators
    ]
    
    for batch_sz, gen_count in test_cases:
        probs = torch.rand(batch_sz, vocab_size, device=device).softmax(dim=-1)
        generators = {i: torch.Generator(device=device).manual_seed(42+i) 
                     for i in range(gen_count)}
        
        # Test multiple runs for statistical consistency
        orig_results = []
        opt_results = []
        
        for run in range(5):
            # Use same seeds for fair comparison
            gen_orig = {k: torch.Generator(device=device).manual_seed(42+k) 
                       for k in generators.keys()}
            gen_opt = {k: torch.Generator(device=device).manual_seed(42+k) 
                      for k in generators.keys()}
            
            orig_results.append(random_sample_original(probs.clone(), gen_orig))
            opt_results.append(random_sample_optimized(probs.clone(), gen_opt))
        
        # Check if results are deterministic with same seeds
        orig_consistent = all(torch.equal(orig_results[0], r) for r in orig_results[1:])
        opt_consistent = all(torch.equal(opt_results[0], r) for r in opt_results[1:])
        
        print(f"   Batch {batch_sz}, Gen {gen_count}:")
        print(f"     Original deterministic: {'✅' if orig_consistent else '❌'}")
        print(f"     Optimized deterministic: {'✅' if opt_consistent else '❌'}")
        print(f"     Results match: {'✅' if torch.equal(orig_results[0], opt_results[0]) else '❌'}")


def benchmark_performance():
    """Comprehensive performance benchmark."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        torch.cuda.reset_peak_memory_stats()
    
    # Test configurations - more comprehensive for GPU
    batch_sizes = [16, 32, 64, 128, 256] if device == 'cuda' else [16, 32, 64]
    vocab_sizes = [32000, 128000] if device == 'cuda' else [32000]
    num_runs = 100 if device == 'cuda' else 50
    
    results = []
    
    for vocab_size in vocab_sizes:
        for batch_size in batch_sizes:
            print(f"\n=== Batch size: {batch_size}, Vocab size: {vocab_size} ===")
            
            # Create test data
            probs = torch.rand(batch_size, vocab_size, device=device)
            probs = probs.softmax(dim=-1)
            
            # Test different generator ratios
            generator_ratios = [0.25, 0.5, 0.75]  # 25%, 50%, 75%
            
            for ratio in generator_ratios:
                gen_count = max(1, int(batch_size * ratio))
                print(f"  With {gen_count}/{batch_size} generators ({ratio*100:.0f}%):")
                
                # Create generators
                generators = {i: torch.Generator(device=device).manual_seed(42+i) 
                             for i in range(gen_count)}
                
                # Warmup
                for _ in range(5):
                    _ = random_sample_original(probs.clone(), generators.copy())
                    _ = random_sample_optimized(probs.clone(), generators.copy())
                
                # Benchmark original
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                for _ in range(num_runs):
                    result_orig = random_sample_original(probs.clone(), generators.copy())
                if device == 'cuda':
                    torch.cuda.synchronize()
                orig_time = time.time() - start
                
                # Benchmark optimized
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                for _ in range(num_runs):
                    result_opt = random_sample_optimized(probs.clone(), generators.copy())
                if device == 'cuda':
                    torch.cuda.synchronize()
                opt_time = time.time() - start
                
                # Calculate metrics
                orig_ms = orig_time / num_runs * 1000
                opt_ms = opt_time / num_runs * 1000
                speedup = orig_time / opt_time
                
                print(f"    Original:  {orig_ms:.3f}ms per call")
                print(f"    Optimized: {opt_ms:.3f}ms per call")
                print(f"    Speedup:   {speedup:.2f}x")
                
                # Determine threshold used
                vectorization_threshold = min(4 + batch_size // 32, 8)
                using_vectorization = gen_count > vectorization_threshold
                print(f"    Vectorization: {'ON' if using_vectorization else 'OFF'} (threshold: {vectorization_threshold})")
                
                # Store results
                results.append({
                    'batch_size': batch_size,
                    'vocab_size': vocab_size,
                    'gen_count': gen_count,
                    'gen_ratio': ratio,
                    'orig_ms': orig_ms,
                    'opt_ms': opt_ms,
                    'speedup': speedup,
                    'vectorization': using_vectorization
                })
                
                if speedup > 1.1:
                    print("    ✅ Significant improvement!")
                elif speedup > 1.05:
                    print("    🟡 Minor improvement")
                elif speedup > 0.95:
                    print("    ⚪ Similar performance")
                else:
                    print("    ❌ Performance regression")
    
    return results


def print_summary(results, device):
    """Print comprehensive summary."""
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    if not results:
        print("No results to summarize.")
        return
    
    # Overall statistics
    effective_cases = [r for r in results if r['speedup'] > 1.1]
    minor_improvements = [r for r in results if 1.05 < r['speedup'] <= 1.1]
    regression_cases = [r for r in results if r['speedup'] < 0.95]
    
    print(f"Total test cases: {len(results)}")
    print(f"Significant improvements (>10%): {len(effective_cases)} ({len(effective_cases)/len(results)*100:.1f}%)")
    print(f"Minor improvements (5-10%): {len(minor_improvements)} ({len(minor_improvements)/len(results)*100:.1f}%)")
    print(f"Performance regressions (>5%): {len(regression_cases)} ({len(regression_cases)/len(results)*100:.1f}%)")
    
    # Best case analysis
    if effective_cases:
        best_case = max(effective_cases, key=lambda x: x['speedup'])
        print(f"\nBest speedup: {best_case['speedup']:.2f}x")
        print(f"  Configuration: Batch={best_case['batch_size']}, Vocab={best_case['vocab_size']}")
        print(f"  Generators: {best_case['gen_count']} ({best_case['gen_ratio']*100:.0f}%)")
        print(f"  Vectorization: {'ON' if best_case['vectorization'] else 'OFF'}")
    
    # Vectorization analysis
    vec_on = [r for r in results if r['vectorization']]
    vec_off = [r for r in results if not r['vectorization']]
    
    if vec_on:
        avg_speedup_vec = sum(r['speedup'] for r in vec_on) / len(vec_on)
        print(f"\nVectorization ON (n={len(vec_on)}): Avg speedup = {avg_speedup_vec:.2f}x")
    
    if vec_off:
        avg_speedup_no_vec = sum(r['speedup'] for r in vec_off) / len(vec_off)
        print(f"Vectorization OFF (n={len(vec_off)}): Avg speedup = {avg_speedup_no_vec:.2f}x")
    
    # Overall metrics
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    median_speedup = sorted(results, key=lambda x: x['speedup'])[len(results)//2]['speedup']
    
    print(f"\nOverall average speedup: {avg_speedup:.2f}x")
    print(f"Median speedup: {median_speedup:.2f}x")
    
    # Memory usage (GPU only)
    if device == 'cuda':
        try:
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            print(f"Peak GPU memory usage: {peak_memory:.2f} GB")
        except:
            pass
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if len(effective_cases) / len(results) > 0.5:
        print("✅ Optimization is generally effective!")
    elif len(regression_cases) / len(results) > 0.3:
        print("⚠️  Optimization has significant regressions. Consider tuning.")
    else:
        print("🟡 Mixed results. Optimization may be beneficial in specific scenarios.")


def main():
    """Main test execution."""
    print("vLLM v1 Sampling Optimization - Comprehensive Test Suite")
    print(f"PyTorch version: {torch.__version__}")
    
    # Run correctness tests first
    test_correctness()
    
    # Run performance benchmarks
    try:
        results = benchmark_performance()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print_summary(results, device)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()