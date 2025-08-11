#!/usr/bin/env python3
"""
Comprehensive stress test for QKV optimization including memory profiling and realistic workloads.
"""
import torch
import time
import gc
import psutil
import os
from typing import List, Dict
# matplotlib is optional for this test
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class StressTestConfig:
    # LLaMA model configurations to test
    LLAMA_CONFIGS = {
        "LLaMA-7B": {"hidden_size": 4096, "num_heads": 32, "num_layers": 32},
        "LLaMA-13B": {"hidden_size": 5120, "num_heads": 40, "num_layers": 40},
        "LLaMA-30B": {"hidden_size": 6656, "num_heads": 52, "num_layers": 60},
        "LLaMA-65B": {"hidden_size": 8192, "num_heads": 64, "num_layers": 80},
    }
    
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    SEQUENCE_LENGTHS = [128, 512, 1024, 2048, 4096]
    WARMUP_ITERATIONS = 100
    BENCHMARK_ITERATIONS = 500

class QKVStressTester:
    def __init__(self, model_config: Dict, device: str = "auto"):
        self.hidden_size = model_config["hidden_size"]
        self.num_heads = model_config["num_heads"]
        self.num_layers = model_config["num_layers"]
        self.head_dim = self.hidden_size // self.num_heads
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Simulate GQA (Grouped Query Attention) - 4x fewer KV heads
        self.num_kv_heads = max(1, self.num_heads // 4)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        # Pre-compute cached split sizes
        self._cached_split_sizes = [self.q_size, self.kv_size, self.kv_size]
        
        print(f"Model: {self.num_layers} layers, {self.hidden_size} hidden, {self.num_heads} heads")
        print(f"Q size: {self.q_size}, KV size: {self.kv_size}")
    
    def create_test_data(self, batch_size: int, seq_len: int):
        """Create test tensor data."""
        qkv_size = self.q_size + 2 * self.kv_size
        return torch.randn(
            batch_size, seq_len, qkv_size,
            device=self.device,
            dtype=torch.float16
        )
    
    def benchmark_original_method(self, test_tensor: torch.Tensor, iterations: int):
        """Benchmark original dynamic list creation method."""
        latencies = []
        
        # Warmup
        for _ in range(50):
            q, k, v = test_tensor.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            del q, k, v
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(iterations):
            if self.device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()
            
            # Original method - create list each time
            q, k, v = test_tensor.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            
            if self.device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event)
            else:
                elapsed = (time.perf_counter() - start_time) * 1000
            
            latencies.append(elapsed)
            del q, k, v
        
        return latencies
    
    def benchmark_optimized_method(self, test_tensor: torch.Tensor, iterations: int):
        """Benchmark optimized cached split sizes method."""
        latencies = []
        
        # Warmup
        for _ in range(50):
            q, k, v = test_tensor.split(self._cached_split_sizes, dim=-1)
            del q, k, v
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(iterations):
            if self.device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()
            
            # Optimized method - use cached sizes
            q, k, v = test_tensor.split(self._cached_split_sizes, dim=-1)
            
            if self.device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event)
            else:
                elapsed = (time.perf_counter() - start_time) * 1000
            
            latencies.append(elapsed)
            del q, k, v
        
        return latencies
    
    def measure_memory_usage(self):
        """Measure current memory usage."""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "gpu_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                "gpu_reserved": torch.cuda.memory_reserved() / 1024**2,    # MB
                "gpu_max_allocated": torch.cuda.max_memory_allocated() / 1024**2,  # MB
            }
        else:
            process = psutil.Process(os.getpid())
            return {
                "cpu_memory": process.memory_info().rss / 1024**2,  # MB
            }
    
    def run_comprehensive_test(self, batch_size: int, seq_len: int, iterations: int):
        """Run comprehensive performance test."""
        print(f"\nTesting Batch={batch_size}, SeqLen={seq_len}")
        
        # Create test data
        test_tensor = self.create_test_data(batch_size, seq_len)
        
        # Measure initial memory
        initial_memory = self.measure_memory_usage()
        
        # Benchmark original method
        print("  Running original method...")
        original_latencies = self.benchmark_original_method(test_tensor, iterations)
        
        # Clear memory and benchmark optimized method
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        print("  Running optimized method...")
        optimized_latencies = self.benchmark_optimized_method(test_tensor, iterations)
        
        # Measure final memory
        final_memory = self.measure_memory_usage()
        
        # Calculate statistics
        import numpy as np
        original_stats = {
            "mean": np.mean(original_latencies),
            "median": np.median(original_latencies),
            "p95": np.percentile(original_latencies, 95),
            "p99": np.percentile(original_latencies, 99),
            "std": np.std(original_latencies),
        }
        
        optimized_stats = {
            "mean": np.mean(optimized_latencies),
            "median": np.median(optimized_latencies),
            "p95": np.percentile(optimized_latencies, 95),
            "p99": np.percentile(optimized_latencies, 99),
            "std": np.std(optimized_latencies),
        }
        
        # Calculate improvements
        improvement = {
            "mean": ((original_stats["mean"] - optimized_stats["mean"]) / original_stats["mean"]) * 100,
            "median": ((original_stats["median"] - optimized_stats["median"]) / original_stats["median"]) * 100,
            "p95": ((original_stats["p95"] - optimized_stats["p95"]) / original_stats["p95"]) * 100,
            "p99": ((original_stats["p99"] - optimized_stats["p99"]) / original_stats["p99"]) * 100,
        }
        
        # Calculate throughput
        tokens_per_iter = batch_size * seq_len
        original_throughput = tokens_per_iter / (original_stats["mean"] / 1000)  # tokens/sec
        optimized_throughput = tokens_per_iter / (optimized_stats["mean"] / 1000)  # tokens/sec
        throughput_improvement = ((optimized_throughput - original_throughput) / original_throughput) * 100
        
        return {
            "config": {"batch_size": batch_size, "seq_len": seq_len},
            "original": original_stats,
            "optimized": optimized_stats,
            "improvement": improvement,
            "throughput_improvement": throughput_improvement,
            "memory": {"initial": initial_memory, "final": final_memory},
            "raw_latencies": {"original": original_latencies, "optimized": optimized_latencies}
        }
    
    def run_stress_test_suite(self):
        """Run comprehensive stress test suite."""
        results = []
        
        print(f"Starting comprehensive stress test...")
        print(f"Device: {self.device}")
        print(f"Model configuration: {self.num_layers} layers, {self.hidden_size} hidden size")
        
        total_tests = len(StressTestConfig.BATCH_SIZES) * len(StressTestConfig.SEQUENCE_LENGTHS)
        current_test = 0
        
        for batch_size in StressTestConfig.BATCH_SIZES:
            for seq_len in StressTestConfig.SEQUENCE_LENGTHS:
                current_test += 1
                print(f"\nProgress: {current_test}/{total_tests}")
                
                try:
                    result = self.run_comprehensive_test(
                        batch_size, seq_len, StressTestConfig.BENCHMARK_ITERATIONS
                    )
                    results.append(result)
                    
                    # Print quick summary
                    print(f"  Improvement: {result['improvement']['mean']:.2f}% (mean latency)")
                    print(f"  Throughput: +{result['throughput_improvement']:.2f}%")
                    
                except Exception as e:
                    print(f"  Error in test: {e}")
                    continue
                
                # Clear memory between tests
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        
        return results

def generate_performance_report(results: List[Dict], model_name: str):
    """Generate comprehensive performance report."""
    if not results:
        print("No results to report!")
        return
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PERFORMANCE REPORT - {model_name}")
    print(f"{'='*80}")
    
    # Overall statistics
    mean_improvements = [r["improvement"]["mean"] for r in results]
    throughput_improvements = [r["throughput_improvement"] for r in results]
    
    import numpy as np
    
    print(f"\nOVERALL PERFORMANCE SUMMARY:")
    print(f"  Average Latency Improvement: {np.mean(mean_improvements):.2f}%")
    print(f"  Best Latency Improvement: {np.max(mean_improvements):.2f}%")
    print(f"  Worst Latency Improvement: {np.min(mean_improvements):.2f}%")
    print(f"  Latency Improvement Std Dev: {np.std(mean_improvements):.2f}%")
    print(f"  Average Throughput Improvement: {np.mean(throughput_improvements):.2f}%")
    print(f"  Best Throughput Improvement: {np.max(throughput_improvements):.2f}%")
    
    # Detailed results table
    print(f"\nDETAILED RESULTS:")
    print(f"{'Batch':<6} {'SeqLen':<7} {'Orig(ms)':<10} {'Opt(ms)':<10} {'Improve':<8} {'Throughput+':<12}")
    print(f"{'-'*60}")
    
    for result in results:
        config = result["config"]
        orig_mean = result["original"]["mean"]
        opt_mean = result["optimized"]["mean"]
        improvement = result["improvement"]["mean"]
        throughput_imp = result["throughput_improvement"]
        
        print(f"{config['batch_size']:<6} {config['seq_len']:<7} {orig_mean:<10.4f} {opt_mean:<10.4f} {improvement:<7.2f}% {throughput_imp:<11.2f}%")
    
    # Find best performing configurations
    best_latency_idx = np.argmax(mean_improvements)
    best_throughput_idx = np.argmax(throughput_improvements)
    
    print(f"\nBEST PERFORMING CONFIGURATIONS:")
    best_latency_config = results[best_latency_idx]["config"]
    best_throughput_config = results[best_throughput_idx]["config"]
    
    print(f"  Best Latency Improvement: Batch={best_latency_config['batch_size']}, SeqLen={best_latency_config['seq_len']} ({mean_improvements[best_latency_idx]:.2f}%)")
    print(f"  Best Throughput Improvement: Batch={best_throughput_config['batch_size']}, SeqLen={best_throughput_config['seq_len']} ({throughput_improvements[best_throughput_idx]:.2f}%)")

def main():
    print("QKV Split Optimization - Comprehensive Stress Test")
    print("="*50)
    
    # Test LLaMA-7B configuration (most commonly used)
    model_config = StressTestConfig.LLAMA_CONFIGS["LLaMA-7B"]
    
    tester = QKVStressTester(model_config)
    results = tester.run_stress_test_suite()
    
    # Generate report
    generate_performance_report(results, "LLaMA-7B")
    
    print(f"\nStress test completed!")
    print(f"Total tests run: {len(results)}")

if __name__ == "__main__":
    main()