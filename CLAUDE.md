# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models. It's designed for production use with features like PagedAttention, continuous batching, and optimized CUDA kernels.

## Development Commands

### Installation and Setup
```bash
# Editable development install
pip install -e .

# Install development dependencies
pip install -r requirements/dev.txt

# Install pre-commit hooks (required for linting)
pre-commit install
```

### Code Quality and Linting
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific linters
yapf --in-place --verbose <file>           # Format Python code
ruff check --fix <file>                   # Lint Python code
isort <file>                              # Sort imports
clang-format -i <file>                    # Format C++/CUDA code

# Type checking
./tools/mypy.sh                           # Run mypy type checker
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/basic_correctness/           # Basic correctness tests
pytest tests/distributed/                # Distributed tests
pytest tests/engine/                      # Engine tests
pytest tests/models/                      # Model tests

# Run tests with markers
pytest -m "not distributed"              # Skip distributed tests
pytest -m "core_model"                   # Run core model tests only
```

### Building and Compilation
```bash
# Clean build (if needed)
rm -rf build/ *.egg-info/

# Build with specific device target
VLLM_TARGET_DEVICE=cuda pip install -e .  # CUDA build
VLLM_TARGET_DEVICE=cpu pip install -e .   # CPU build
VLLM_TARGET_DEVICE=rocm pip install -e .  # ROCm build

# Build with optimizations
CMAKE_BUILD_TYPE=Release pip install -e .
```

### Documentation
```bash
# Serve documentation locally (if mkdocs is available)
mkdocs serve
```

## Code Architecture

### Core Components

1. **Engine Layer** (`vllm/engine/`)
   - `LLMEngine`: Main inference engine with request scheduling
   - `AsyncLLMEngine`: Asynchronous wrapper for concurrent requests
   - `arg_utils.py`: Configuration and argument parsing

2. **Model Executor** (`vllm/model_executor/`)
   - `models/`: 100+ supported model implementations
   - `layers/`: Custom optimized neural network layers
   - `model_loader.py`: Efficient model loading strategies
   - `quantization/`: Multiple quantization backends (GPTQ, AWQ, FP8)

3. **Attention System** (`vllm/attention/`)
   - `backends/`: FlashAttention, PagedAttention, FlashInfer implementations
   - `ops/`: CUDA/HIP optimized attention kernels
   - `selector.py`: Automatic backend selection

4. **Distributed Computing** (`vllm/distributed/`)
   - `communication/`: Custom all-reduce operations
   - `device_communicators/`: Platform-specific communication
   - `parallel_state.py`: Distributed execution state

5. **Scheduling** (`vllm/core/`)
   - `scheduler.py`: Request batching and scheduling logic
   - `block_manager.py`: Memory allocation and management
   - `interfaces.py`: Core abstractions

6. **Entrypoints** (`vllm/entrypoints/`)
   - `llm.py`: Simple Python API
   - `openai/api_server.py`: OpenAI-compatible REST API
   - `cli/`: Command-line interface

### Key Design Patterns

- **PagedAttention**: Memory-efficient attention with virtual memory management
- **Continuous Batching**: Dynamic batching of requests for optimal throughput
- **Quantization**: Multiple precision formats (FP16, FP8, INT4, INT8)
- **Multi-modal Support**: Text, vision, and audio processing
- **Distributed Inference**: Tensor, pipeline, and expert parallelism

## Development Guidelines

### Code Style
- Python code uses yapf formatter with 80-character line limit
- C++/CUDA code uses clang-format
- All code must pass pre-commit hooks before committing
- Use type hints for new Python code

### Testing Requirements
- Write unit tests for new utilities and core functionality
- Add integration tests for new model implementations
- Include distributed tests for multi-GPU features
- Test on both CUDA and CPU backends when applicable

### Model Integration
- New models go in `vllm/model_executor/models/`
- Follow existing model implementation patterns
- Register models in `vllm/model_executor/models/registry.py`
- Add corresponding tests in `tests/models/`

### Performance Considerations
- Profile new features with `vllm/utils/profiler.py`
- Optimize memory usage - vLLM is memory-constrained
- Use CUDA graphs for performance-critical paths
- Consider quantization support for new operations

## Platform Support

- **Primary**: NVIDIA GPUs with CUDA
- **Secondary**: AMD GPUs with ROCm, Intel GPUs (XPU), CPUs
- **Experimental**: TPUs, AWS Neuron, Intel Habana

## Common Issues

- **Memory**: Use `--max-model-len` to limit context length if OOM
- **Compilation**: Clean build directory if seeing linker errors
- **Performance**: Enable CUDA graphs with `--enforce-eager=False`
- **Quantization**: Check GPU compute capability for FP8/INT8 support

## Environment Variables

Key environment variables that affect development:
- `VLLM_TARGET_DEVICE`: Target device (cuda, cpu, rocm)
- `CUDA_HOME`: CUDA installation path
- `MAX_JOBS`: Parallel compilation jobs
- `VLLM_USE_PRECOMPILED`: Use precompiled binaries
- `CMAKE_BUILD_TYPE`: Build type (Debug, Release, RelWithDebInfo)