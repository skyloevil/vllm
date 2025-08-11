#!/usr/bin/env python3
"""
Benchmark script to measure performance improvement of QKV split optimization in LLaMA model.
"""
import argparse
import time
import torch
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    memory_usage: float

class QKVSplitBenchmark:
    def __init__(self, batch_size: int, seq_len: int, hidden_size: int, num_heads: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Simulate QKV projection output
        self.qkv_size = hidden_size + 2 * (hidden_size // 4)  # Assuming GQA with 4x fewer KV heads
        self.q_size = hidden_size
        self.kv_size = hidden_size // 4
        
        # Create test tensor
        self.test_tensor = torch.randn(
            batch_size, seq_len, self.qkv_size, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype=torch.float16
        )
        
        # Pre-computed split sizes (optimized version)
        self.cached_split_sizes = [self.q_size, self.kv_size, self.kv_size]
    
    def benchmark_original(self, num_iterations: int = 1000) -> BenchmarkResult:
        """Benchmark the original implementation with dynamic list creation."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        latencies = []
        
        # Warmup
        for _ in range(50):
            q, k, v = self.test_tensor.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            del q, k, v
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Actual benchmark
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
            else:
                start_time = time.perf_counter()
            
            # Original implementation - create list every time
            q, k, v = self.test_tensor.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            del q, k, v
        
        return self._calculate_metrics(latencies)
    
    def benchmark_optimized(self, num_iterations: int = 1000) -> BenchmarkResult:
        """Benchmark the optimized implementation with cached split sizes."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        latencies = []
        
        # Warmup
        for _ in range(50):
            q, k, v = self.test_tensor.split(self.cached_split_sizes, dim=-1)
            del q, k, v
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Actual benchmark
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
            else:
                start_time = time.perf_counter()
            
            # Optimized implementation - use cached split sizes
            q, k, v = self.test_tensor.split(self.cached_split_sizes, dim=-1)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            del q, k, v
        
        return self._calculate_metrics(latencies)
    
    def _calculate_metrics(self, latencies: List[float]) -> BenchmarkResult:
        """Calculate benchmark metrics from latency measurements."""
        latencies = np.array(latencies)
        
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Calculate throughput (tokens per second)
        tokens_per_batch = self.batch_size * self.seq_len
        throughput = tokens_per_batch / (avg_latency / 1000)  # Convert ms to seconds
        
        # Estimate memory usage (MB)
        tensor_size_mb = self.test_tensor.numel() * self.test_tensor.element_size() / (1024 * 1024)
        memory_usage = tensor_size_mb * 4  # Rough estimate including intermediate tensors
        
        return BenchmarkResult(
            avg_latency=avg_latency,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput=throughput,
            memory_usage=memory_usage
        )

def print_results(original: BenchmarkResult, optimized: BenchmarkResult, config: str):
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"Benchmark Results - {config}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
    print(f"{'-'*60}")
    
    # Latency metrics (lower is better)
    avg_improvement = ((original.avg_latency - optimized.avg_latency) / original.avg_latency) * 100
    print(f"{'Avg Latency (ms)':<20} {original.avg_latency:<15.4f} {optimized.avg_latency:<15.4f} {avg_improvement:<14.2f}%")
    
    p50_improvement = ((original.p50_latency - optimized.p50_latency) / original.p50_latency) * 100
    print(f"{'P50 Latency (ms)':<20} {original.p50_latency:<15.4f} {optimized.p50_latency:<15.4f} {p50_improvement:<14.2f}%")
    
    p95_improvement = ((original.p95_latency - optimized.p95_latency) / original.p95_latency) * 100
    print(f"{'P95 Latency (ms)':<20} {original.p95_latency:<15.4f} {optimized.p95_latency:<15.4f} {p95_improvement:<14.2f}%")
    
    p99_improvement = ((original.p99_latency - optimized.p99_latency) / original.p99_latency) * 100
    print(f"{'P99 Latency (ms)':<20} {original.p99_latency:<15.4f} {optimized.p99_latency:<15.4f} {p99_improvement:<14.2f}%")
    
    # Throughput metrics (higher is better)
    throughput_improvement = ((optimized.throughput - original.throughput) / original.throughput) * 100
    print(f"{'Throughput (tok/s)':<20} {original.throughput:<15.2f} {optimized.throughput:<15.2f} {throughput_improvement:<14.2f}%")
    
    print(f"{'Memory Usage (MB)':<20} {original.memory_usage:<15.2f} {optimized.memory_usage:<15.2f} {'~0.00%':<15}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark QKV split optimization")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of benchmark iterations")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16], help="Batch sizes to test")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 1024, 2048], help="Sequence lengths to test")
    args = parser.parse_args()
    
    print(f"Running QKV Split Optimization Benchmark")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Iterations per test: {args.iterations}")
    
    # Standard LLaMA-2 7B configuration
    hidden_size = 4096
    num_heads = 32
    
    total_improvements = []
    
    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            config = f"Batch={batch_size}, SeqLen={seq_len}"
            
            benchmark = QKVSplitBenchmark(batch_size, seq_len, hidden_size, num_heads)
            
            print(f"\nRunning benchmark for {config}...")
            original_result = benchmark.benchmark_original(args.iterations)
            optimized_result = benchmark.benchmark_optimized(args.iterations)
            
            print_results(original_result, optimized_result, config)
            
            # Track overall improvement
            improvement = ((original_result.avg_latency - optimized_result.avg_latency) / original_result.avg_latency) * 100
            total_improvements.append(improvement)
    
    # Overall summary
    print(f"\n{'='*60}")
    print(f"Overall Performance Summary")
    print(f"{'='*60}")
    print(f"Average Improvement: {np.mean(total_improvements):.2f}%")
    print(f"Best Improvement: {np.max(total_improvements):.2f}%")
    print(f"Worst Improvement: {np.min(total_improvements):.2f}%")
    print(f"Standard Deviation: {np.std(total_improvements):.2f}%")

if __name__ == "__main__":
    main()