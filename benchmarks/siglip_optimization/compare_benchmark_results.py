#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare benchmark results between baseline and optimized branches."""

import sys
from pathlib import Path
from typing import Optional

import regex as re


def extract_metric(content: str, pattern: str, group_idx: int = 1) -> Optional[float]:
    """Extract a metric using regex pattern."""
    match = re.search(pattern, content, re.MULTILINE)
    if match:
        try:
            return float(match.group(group_idx))
        except (ValueError, IndexError):
            return None
    return None


def parse_benchmark_file(filepath: Path) -> dict[str, Optional[float]]:
    """Parse benchmark output file and extract key metrics."""
    content = filepath.read_text()

    metrics = {}

    # Throughput metrics
    metrics["req_throughput"] = extract_metric(
        content, r"Request throughput:\s+([\d.]+)\s+req/s"
    )
    metrics["token_throughput"] = extract_metric(
        content, r"Output token throughput:\s+([\d.]+)\s+tok/s"
    )
    metrics["peak_token_throughput"] = extract_metric(
        content, r"Peak output token throughput:\s+([\d.]+)\s+tok/s"
    )

    # TTFT metrics (Time to First Token)
    metrics["ttft_mean"] = extract_metric(content, r"TTFT.*?Mean:\s+([\d.]+)\s+ms")
    metrics["ttft_median"] = extract_metric(content, r"TTFT.*?Median:\s+([\d.]+)\s+ms")
    metrics["ttft_p99"] = extract_metric(content, r"TTFT.*?P99:\s+([\d.]+)\s+ms")

    # TPOT metrics (Time per Output Token)
    metrics["tpot_mean"] = extract_metric(content, r"TPOT.*?Mean:\s+([\d.]+)\s+ms")
    metrics["tpot_median"] = extract_metric(content, r"TPOT.*?Median:\s+([\d.]+)\s+ms")
    metrics["tpot_p99"] = extract_metric(content, r"TPOT.*?P99:\s+([\d.]+)\s+ms")

    # Request count
    metrics["total_requests"] = extract_metric(content, r"Total requests:\s+(\d+)")

    # Duration
    metrics["duration"] = extract_metric(content, r"Duration:\s+([\d.]+)\s+s")

    return metrics


def calculate_improvement(baseline: float, optimized: float) -> float:
    """Calculate percentage improvement."""
    if baseline == 0:
        return 0.0
    return ((optimized - baseline) / baseline) * 100


def calculate_reduction(baseline: float, optimized: float) -> float:
    """Calculate percentage reduction (for latency metrics)."""
    if baseline == 0:
        return 0.0
    return ((baseline - optimized) / baseline) * 100


def format_improvement(value: float, is_latency: bool = False) -> str:
    """Format improvement percentage with color."""
    if is_latency:
        # For latency, lower is better
        if value > 0:
            return f"-{value:.2f}%"
        elif value < 0:
            return f"+{abs(value):.2f}%"
        else:
            return "0.00%"
    else:
        # For throughput, higher is better
        if value > 0:
            return f"+{value:.2f}%"
        elif value < 0:
            return f"{value:.2f}%"
        else:
            return "0.00%"


def print_comparison_table(
    main_metrics: dict[str, Optional[float]],
    opt_metrics: dict[str, Optional[float]],
) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPARISON: Main (Bicubic) vs Optimized (Bilinear)")
    print("=" * 80)

    # Throughput Metrics
    print("\nTHROUGHPUT METRICS\n")
    print(f"{'Metric':<30} {'Main':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 80)

    throughput_metrics = [
        ("Request throughput (req/s)", "req_throughput", False),
        ("Token throughput (tok/s)", "token_throughput", False),
        ("Peak token throughput (tok/s)", "peak_token_throughput", False),
    ]

    for label, key, _ in throughput_metrics:
        main_val = main_metrics.get(key)
        opt_val = opt_metrics.get(key)

        if main_val is not None and opt_val is not None:
            improvement = calculate_improvement(main_val, opt_val)
            print(
                f"{label:<30} {main_val:<15.2f} {opt_val:<15.2f} "
                f"{format_improvement(improvement)}"
            )
        else:
            print(f"{label:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

    # Latency Metrics
    print("\nLATENCY METRICS\n")
    print(f"{'Metric':<30} {'Main':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 80)

    latency_metrics = [
        ("TTFT Mean (ms)", "ttft_mean", True),
        ("TTFT Median (ms)", "ttft_median", True),
        ("TTFT P99 (ms)", "ttft_p99", True),
        ("TPOT Mean (ms)", "tpot_mean", True),
        ("TPOT Median (ms)", "tpot_median", True),
        ("TPOT P99 (ms)", "tpot_p99", True),
    ]

    for label, key, is_latency in latency_metrics:
        main_val = main_metrics.get(key)
        opt_val = opt_metrics.get(key)

        if main_val is not None and opt_val is not None:
            reduction = calculate_reduction(main_val, opt_val)
            print(
                f"{label:<30} {main_val:<15.2f} {opt_val:<15.2f} "
                f"{format_improvement(reduction, is_latency=True)}"
            )
        else:
            print(f"{label:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

    # Summary Statistics
    print("\nSUMMARY\n")
    print("-" * 80)

    total_main = main_metrics.get("total_requests", 0)
    total_opt = opt_metrics.get("total_requests", 0)
    duration_main = main_metrics.get("duration", 0)
    duration_opt = opt_metrics.get("duration", 0)

    print(f"Total requests:        Main: {total_main:.0f}, Optimized: {total_opt:.0f}")
    print(
        "Benchmark duration:    Main: "
        f"{duration_main:.2f}s, Optimized: {duration_opt:.2f}s"
    )

    # Calculate key improvements
    req_throughput_improvement = 0
    ttft_mean_reduction = 0

    if main_metrics.get("req_throughput") and opt_metrics.get("req_throughput"):
        req_throughput_improvement = calculate_improvement(
            main_metrics["req_throughput"], opt_metrics["req_throughput"]
        )

    if main_metrics.get("ttft_mean") and opt_metrics.get("ttft_mean"):
        ttft_mean_reduction = calculate_reduction(
            main_metrics["ttft_mean"], opt_metrics["ttft_mean"]
        )

    print("\nKEY IMPROVEMENTS\n")
    print(f"  - Request throughput: {format_improvement(req_throughput_improvement)}")
    ttft_string = format_improvement(
        ttft_mean_reduction,
        is_latency=True,
    )
    print(f"  - TTFT (mean):        {ttft_string}")

    # Verdict
    print("\n" + "=" * 80)
    if req_throughput_improvement > 1 or ttft_mean_reduction > 1:
        print("OPTIMIZATION EFFECTIVE: Measurable performance improvement detected!")
    elif req_throughput_improvement > 0 or ttft_mean_reduction > 0:
        print("MARGINAL IMPROVEMENT: Small performance gains observed.")
    else:
        print("NO IMPROVEMENT: Consider investigating the optimization.")
    print("=" * 80 + "\n")


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python compare_benchmark_results.py <main_results.txt> "
            "<optimized_results.txt>"
        )
        sys.exit(1)

    main_file = Path(sys.argv[1])
    opt_file = Path(sys.argv[2])

    if not main_file.exists():
        print(f"Error: File not found: {main_file}")
        sys.exit(1)

    if not opt_file.exists():
        print(f"Error: File not found: {opt_file}")
        sys.exit(1)

    print("\nAnalyzing benchmark results...")
    print(f"   Main:      {main_file}")
    print(f"   Optimized: {opt_file}")

    main_metrics = parse_benchmark_file(main_file)
    opt_metrics = parse_benchmark_file(opt_file)

    print_comparison_table(main_metrics, opt_metrics)

    # Generate markdown table for PR
    print("\nMARKDOWN TABLE FOR PR\n")
    print("```markdown")
    print("| Metric | Main (Bicubic) | Optimized (Bilinear) | Change |")
    print("|--------|----------------|----------------------|--------|")

    for label, key, is_latency in [
        ("Request throughput (req/s)", "req_throughput", False),
        ("Token throughput (tok/s)", "token_throughput", False),
        ("TTFT Mean (ms)", "ttft_mean", True),
        ("TTFT P99 (ms)", "ttft_p99", True),
        ("TPOT Mean (ms)", "tpot_mean", True),
    ]:
        main_val = main_metrics.get(key)
        opt_val = opt_metrics.get(key)

        if main_val is not None and opt_val is not None:
            if is_latency:
                change = calculate_reduction(main_val, opt_val)
                change_str = f"-{change:.1f}%" if change > 0 else f"+{abs(change):.1f}%"
            else:
                change = calculate_improvement(main_val, opt_val)
                change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"

            print(f"| {label} | {main_val:.2f} | {opt_val:.2f} | {change_str} |")

    print("```\n")


if __name__ == "__main__":
    main()
