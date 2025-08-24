# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for optimized FP8 block casting operations."""

import torch

from vllm.triton_utils import DEFAULT_MAX_BLOCK_SIZE, tl, triton
from vllm.utils.deep_gemm import DEFAULT_BLOCK_SIZE, _align


@triton.jit
def _per_block_fp8_cast_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_ue8m0: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized Triton kernel for per-block FP8 casting.
    
    This kernel performs block-wise FP8 conversion with optimized memory access
    patterns and fused operations, replacing the PyTorch implementation.
    """
    # Get program IDs for 2D grid
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Calculate block boundaries
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create offset arrays for memory access
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    # Create 2D offset grid
    offs_m = tl.expand_dims(offs_m, axis=1)
    offs_n = tl.expand_dims(offs_n, axis=0)
    offs = offs_m * N + offs_n

    # Create masks for boundary checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m & mask_n

    # Load input data
    input_data = tl.load(input_ptr + offs, mask=mask, other=0.0)

    # Compute absolute values
    abs_data = tl.abs(input_data)

    # Find maximum value in the block
    block_max = tl.max(abs_data, axis=1)
    block_max = tl.max(block_max, axis=0)

    # Clamp minimum value to avoid division by zero
    block_max = tl.maximum(block_max, 1e-4)

    # Compute scaling factor (448.0 is FP8 E4M3 max representable value)
    scale = block_max / 448.0

    # Apply UE8M0 scaling if requested
    if use_ue8m0:
        # Equivalent to _ceil_to_ue8m0: ceil to next power of 2
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    # Apply scaling and convert to FP8
    scaled_data = input_data * (1.0 / scale)

    # Store results
    tl.store(output_ptr + offs, scaled_data, mask=mask)

    # Store scale factor (one per block)
    scale_offset = (pid_m * tl.cdiv(N, block_n) + pid_n)
    tl.store(scale_ptr + scale_offset, scale)


def per_block_cast_to_fp8_triton(
        x: torch.Tensor,
        block_size: list[int] = DEFAULT_BLOCK_SIZE,
        use_ue8m0: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized Triton implementation of per-block FP8 casting.
    
    This function replaces the PyTorch-based implementation with a fused
    Triton kernel for better performance, especially for large tensors.
    
    Args:
        x: Input tensor to convert (2D)
        block_size: Block dimensions [block_m, block_n]
        use_ue8m0: Whether to use UE8M0 scaling
        
    Returns:
        Tuple of (converted_tensor, scale_factors)
    """
    assert x.dim() == 2, "Input tensor must be 2D"

    m, n = x.shape
    block_m, block_n = block_size

    # Calculate padded dimensions
    padded_m = _align(m, block_m)
    padded_n = _align(n, block_n)

    # Allocate output tensors
    output = torch.zeros((padded_m, padded_n),
                         dtype=torch.float8_e4m3fn,
                         device=x.device)

    # Calculate number of blocks for scale factors
    num_blocks_m = padded_m // block_m
    num_blocks_n = padded_n // block_n
    scale_factors = torch.zeros((num_blocks_m, num_blocks_n),
                                dtype=x.dtype,
                                device=x.device)

    # Pad input tensor if necessary
    if padded_m != m or padded_n != n:
        x_padded = torch.zeros((padded_m, padded_n),
                               dtype=x.dtype,
                               device=x.device)
        x_padded[:m, :n] = x
    else:
        x_padded = x

    # Define grid dimensions
    BLOCK_SIZE_M = min(block_m, DEFAULT_MAX_BLOCK_SIZE)
    BLOCK_SIZE_N = min(block_n, DEFAULT_MAX_BLOCK_SIZE)

    grid_m = triton.cdiv(padded_m, BLOCK_SIZE_M)
    grid_n = triton.cdiv(padded_n, BLOCK_SIZE_N)
    grid = (grid_m, grid_n)

    # Launch kernel
    _per_block_fp8_cast_kernel[grid](
        input_ptr=x_padded,
        output_ptr=output,
        scale_ptr=scale_factors,
        M=padded_m,
        N=padded_n,
        block_m=block_m,
        block_n=block_n,
        use_ue8m0=use_ue8m0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    # Return only the original size portion if we padded
    if padded_m != m or padded_n != n:
        output = output[:m, :n].contiguous()

    return output, scale_factors


def per_block_cast_to_fp8_optimized(
        x: torch.Tensor,
        block_size: list[int] = DEFAULT_BLOCK_SIZE,
        use_ue8m0: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized version choosing between Triton and PyTorch implementations.
    
    This function automatically selects the best implementation based on
    tensor size and hardware capabilities.
    """
    # Use Triton kernel for larger tensors where the overhead is justified
    min_size_for_triton = 1024 * 1024  # 1M elements

    if (x.numel() >= min_size_for_triton and hasattr(triton, 'cdiv')
            and x.device.type == 'cuda'):  # Check Triton availability and CUDA
        return per_block_cast_to_fp8_triton(x, block_size, use_ue8m0)
    else:
        # Fallback to original PyTorch implementation for smaller tensors
        # or when Triton is not available
        from vllm.utils.deep_gemm import per_block_cast_to_fp8
        return per_block_cast_to_fp8(x, block_size, use_ue8m0)
