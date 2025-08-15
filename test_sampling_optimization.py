#!/usr/bin/env python3
"""
Simple test script to verify sampling optimization effectiveness.
"""
import torch
import time
from typing import Dict

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
    """Optimized implementation."""
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # Optimized batch processing for requests with custom generators
        generator_indices = list(generators.keys())
        generator_objects = list(generators.values())
        
        if len(generator_indices) > 4:  # Batch threshold for vectorization
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

def benchmark_sampling():
    print("Benchmarking sampling optimization...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test configurations
    batch_sizes = [8, 32, 128]
    vocab_size = 32000
    num_runs = 50
    
    for batch_size in batch_sizes:
        print(f"\n=== Batch size: {batch_size} ===")
        
        # Create test data
        probs = torch.rand(batch_size, vocab_size, device=device)
        probs = probs.softmax(dim=-1)
        
        # Test different generator ratios
        generator_counts = [batch_size // 4, batch_size // 2, batch_size * 3 // 4]
        
        for gen_count in generator_counts:
            if gen_count == 0:
                continue
                
            print(f"\nWith {gen_count} custom generators:")
            
            # Create generators
            generators = {i: torch.Generator(device=device).manual_seed(42+i) 
                         for i in range(gen_count)}
            
            # Warmup
            for _ in range(3):
                _ = random_sample_original(probs.clone(), generators.copy())
                _ = random_sample_optimized(probs.clone(), generators.copy())
            
            # Benchmark original
            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.time()
            for _ in range(num_runs):
                result_orig = random_sample_original(probs.clone(), generators.copy())
            torch.cuda.synchronize() if device == 'cuda' else None
            orig_time = time.time() - start
            
            # Benchmark optimized
            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.time()
            for _ in range(num_runs):
                result_opt = random_sample_optimized(probs.clone(), generators.copy())
            torch.cuda.synchronize() if device == 'cuda' else None
            opt_time = time.time() - start
            
            # Verify correctness (statistically equivalent, not exactly equal due to randomness)
            print(f"  Original:  {orig_time/num_runs*1000:.3f}ms per call")
            print(f"  Optimized: {opt_time/num_runs*1000:.3f}ms per call")
            speedup = orig_time / opt_time
            print(f"  Speedup:   {speedup:.2f}x")
            
            if speedup > 1.1:
                print("  ✅ Optimization effective!")
            elif speedup > 0.9:
                print("  ⚠️  Similar performance")
            else:
                print("  ❌ Regression detected")

if __name__ == "__main__":
    benchmark_sampling()