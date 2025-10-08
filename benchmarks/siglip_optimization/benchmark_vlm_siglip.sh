#!/bin/bash
# SigLIP Optimization Benchmark - VisionArena Style (10 QPS)
# Replicates PR #25337 methodology for Qwen3-VL

set -e

# Configuration 
MODEL="${MODEL:-google/paligemma-3b-mix-448}"
PORT="${PORT:-8100}"
RESULTS_DIR="./benchmark_results_$(date +%Y%m%d_%H%M%S)"
QPS="${QPS:-10}"  # 10 Queries Per Second (VisionArena standard)
NUM_PROMPTS="${NUM_PROMPTS:-100}"  # Minimum 100 for statistical significance
DATASET="${DATASET:-lmarena-ai/VisionArena-Chat}"  # VisionArena dataset

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. GPU is required for this benchmark."
        exit 1
    fi

    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -eq 0 ]; then
        log_error "No GPU detected."
        exit 1
    fi

    log_info "Found $GPU_COUNT GPU(s)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

wait_for_server() {
    local max_attempts=60
    local attempt=0

    log_info "Waiting for vLLM server to start on port $PORT..."

    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            log_info "Server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done

    log_error "Server failed to start within ${max_attempts} attempts"
    return 1
}

stop_server() {
    log_info "Stopping vLLM server..."
    pkill -f "vllm serve" || true
    sleep 5
}

start_gpu_monitor() {
    local output_file=$1
    log_info "Starting GPU monitoring..."
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv -l 1 > "$output_file" 2>&1 &
    echo $!
}

stop_gpu_monitor() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log_info "Stopping GPU monitor (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

run_vlm_benchmark() {
    local branch=$1
    local output_file=$2
    local gpu_monitor_file=$3

    log_step "Running VLM benchmark for $branch branch..."
    log_info "Dataset: $DATASET"
    log_info "QPS: $QPS (VisionArena standard)"
    log_info "Num prompts: $NUM_PROMPTS"

    # Start GPU monitoring
    GPU_MONITOR_PID=$(start_gpu_monitor "$gpu_monitor_file")

    # Run benchmark with VisionArena dataset
    # Note: --request-rate for controlled QPS (like VisionArena)
    vllm bench serve \
        --backend openai \
        --model "$MODEL" \
        --base-url "http://localhost:$PORT" \
        --dataset-name vision-arena \
        --dataset-path "$DATASET" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$QPS" \
        --save-result \
        --result-dir "$RESULTS_DIR" \
        --result-filename "${branch}_benchmark.json" \
        2>&1 | tee "$output_file"

    local exit_code=${PIPESTATUS[0]}

    # Stop GPU monitoring
    stop_gpu_monitor "$GPU_MONITOR_PID"

    if [ $exit_code -ne 0 ]; then
        log_error "Benchmark failed with exit code $exit_code"
        return 1
    fi

    log_info "Benchmark completed. Results saved to $output_file"
    return 0
}

extract_key_metrics() {
    local file=$1
    local label=$2

    echo ""
    echo "=== $label ==="
    echo ""

    # Request throughput
    grep -E "Request throughput" "$file" || true

    # Output token throughput
    grep -E "Output token throughput" "$file" || true

    echo ""

    # TTFT (Time to First Token) - Key metric for vision encoder performance
    echo "Time to First Token (TTFT):"
    grep -A 5 "TTFT" "$file" | grep -E "Mean|Median|P99" || true

    echo ""

    # TPOT (Time per Output Token)
    echo "Time per Output Token (TPOT):"
    grep -A 5 "TPOT" "$file" | grep -E "Mean|Median|P99" || true

    echo ""
}

summarize_gpu_usage() {
    local gpu_file=$1
    local label=$2

    if [ ! -f "$gpu_file" ]; then
        log_warn "GPU monitoring file not found: $gpu_file"
        return
    fi

    echo "=== GPU Utilization - $label ==="

    # Skip header and calculate averages
    tail -n +2 "$gpu_file" | awk -F',' '
    NR > 0 {
        gpu_util += $2
        mem_util += $3
        mem_used += $4
        count++
    }
    END {
        if (count > 0) {
            printf "Average GPU Utilization: %.1f%%\n", gpu_util/count
            printf "Average Memory Utilization: %.1f%%\n", mem_util/count
            printf "Average Memory Used: %.1f MB\n", mem_used/count
        }
    }'

    echo ""
}

# Main script
main() {
    echo "========================================================"
    echo "  SigLIP Position Encoding Optimization Benchmark"
    echo "  VisionArena Methodology (10 QPS on A100)"
    echo "========================================================"
    echo ""
    echo "Following PR #25337 (Qwen3-VL fast_pos_embed_interpolate)"
    echo ""
    echo "Configuration:"
    echo "  Model:        $MODEL"
    echo "  Port:         $PORT"
    echo "  Dataset:      $DATASET"
    echo "  QPS:          $QPS (controlled request rate)"
    echo "  Num prompts:  $NUM_PROMPTS"
    echo "  Results dir:  $RESULTS_DIR"
    echo ""

    # Check prerequisites
    check_gpu
    echo ""

    # Create results directory
    mkdir -p "$RESULTS_DIR"

    # Save git info
    git rev-parse HEAD > "$RESULTS_DIR/commit_hash.txt"
    git status --short > "$RESULTS_DIR/git_status.txt"

    # Save current branch
    ORIGINAL_BRANCH=$(git branch --show-current)
    log_info "Current branch: $ORIGINAL_BRANCH"
    echo ""

    # Test 1: Main branch (bicubic interpolation)
    log_step "========================================="
    log_step "Test 1: Main branch (bicubic)"
    log_step "========================================="
    echo ""

    git checkout main
    git pull origin main || log_warn "Failed to pull latest main"

    log_info "Starting vLLM server on main branch..."
    vllm serve "$MODEL" \
        --port "$PORT" \
        --trust-remote-code \
        > "$RESULTS_DIR/server_main.log" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID" > "$RESULTS_DIR/server_main_pid.txt"

    if ! wait_for_server; then
        log_error "Failed to start server. Check $RESULTS_DIR/server_main.log"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi

    log_info "Warming up server with 5 requests..."
    sleep 10

    if run_vlm_benchmark "main" \
        "$RESULTS_DIR/benchmark_main.txt" \
        "$RESULTS_DIR/gpu_main.csv"; then
        log_info "Main branch benchmark complete"
    else
        log_error "Main branch benchmark failed"
        stop_server
        exit 1
    fi

    stop_server

    # Test 2: Optimized branch (bilinear interpolation)
    log_step "========================================="
    log_step "Test 2: Optimized branch (bilinear)"
    log_step "========================================="
    echo ""

    git checkout feature/optimize-siglip-pos-interpolation

    log_info "Starting vLLM server on optimized branch..."
    vllm serve "$MODEL" \
        --port "$PORT" \
        --trust-remote-code \
        > "$RESULTS_DIR/server_optimized.log" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID" > "$RESULTS_DIR/server_optimized_pid.txt"

    if ! wait_for_server; then
        log_error "Failed to start server. Check $RESULTS_DIR/server_optimized.log"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi

    log_info "Warming up server with 5 requests..."
    sleep 10

    if run_vlm_benchmark "optimized" \
        "$RESULTS_DIR/benchmark_optimized.txt" \
        "$RESULTS_DIR/gpu_optimized.csv"; then
        log_info "Optimized branch benchmark complete"
    else
        log_error "Optimized branch benchmark failed"
        stop_server
        exit 1
    fi

    stop_server

    # Return to original branch
    git checkout "$ORIGINAL_BRANCH"

    # Display results
    echo ""
    echo "========================================================"
    echo "  BENCHMARK RESULTS"
    echo "========================================================"
    echo ""

    extract_key_metrics "$RESULTS_DIR/benchmark_main.txt" "Main Branch (Bicubic)"
    extract_key_metrics "$RESULTS_DIR/benchmark_optimized.txt" "Optimized Branch (Bilinear)"

    echo ""
    echo "========================================================"
    echo "  GPU UTILIZATION SUMMARY"
    echo "========================================================"
    echo ""

    summarize_gpu_usage "$RESULTS_DIR/gpu_main.csv" "Main Branch"
    summarize_gpu_usage "$RESULTS_DIR/gpu_optimized.csv" "Optimized Branch"

    echo ""
    echo "========================================================"
    echo "  ANALYSIS"
    echo "========================================================"
    echo ""

    # Run Python comparison tool if available
    if [ -f "compare_benchmark_results.py" ]; then
        log_info "Running detailed comparison analysis..."
        python compare_benchmark_results.py \
            "$RESULTS_DIR/benchmark_main.txt" \
            "$RESULTS_DIR/benchmark_optimized.txt"
    fi

    echo ""
    echo "========================================================"
    log_info "Benchmark complete!"
    log_info "Full results available in: $RESULTS_DIR/"
    echo "========================================================"
    echo ""
    echo "Key files:"
    echo "  - benchmark_main.txt - Main branch results"
    echo "  - benchmark_optimized.txt - Optimized branch results"
    echo "  - main_benchmark.json - Structured results (main)"
    echo "  - optimized_benchmark.json - Structured results (optimized)"
    echo "  - gpu_main.csv - GPU utilization (main)"
    echo "  - gpu_optimized.csv - GPU utilization (optimized)"
    echo "  - server_*.log - Server logs"
    echo ""
    echo "Next steps:"
    echo "  1. Review TTFT improvements (should show 10-15% reduction)"
    echo "  2. Verify GPU utilization is similar between branches"
    echo "  3. Add results to PR description"
    echo ""
    echo "Expected improvements (based on PR #25337):"
    echo "  - TTFT Mean: -10 to -15%"
    echo "  - TPOT Mean: -8 to -12%"
    echo "  - Request throughput: +5 to +10%"
    echo ""
}

# Trap to ensure cleanup
trap 'stop_server; git checkout "$ORIGINAL_BRANCH" 2>/dev/null || true' EXIT INT TERM

# Run main
main "$@"
