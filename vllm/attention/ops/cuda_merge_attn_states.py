# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA kernel implementation for attention states merging.

This module provides a CUDA-based alternative to the Triton implementation
for merging attention states, offering reduced CPU overhead and improved
performance for the attention state merging operation.
"""

from typing import Optional

import torch

try:
    import vllm._custom_ops as ops
    _CUDA_KERNELS_AVAILABLE = True
except ImportError:
    _CUDA_KERNELS_AVAILABLE = False


def merge_attn_states_cuda(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    """CUDA-accelerated attention state merging with reduced CPU overhead.
    
    This function implements the same algorithm as the Triton version but
    uses CUDA kernels for better performance and lower CPU overhead.
    
    Args:
        output: Output tensor to store merged results 
            [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        prefix_output: Prefix attention output 
            [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] 
        prefix_lse: Prefix LSE values [NUM_HEADS, NUM_TOKENS]
        suffix_output: Suffix attention output 
            [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        suffix_lse: Suffix LSE values [NUM_HEADS, NUM_TOKENS]
        output_lse: Optional output LSE tensor [NUM_HEADS, NUM_TOKENS]
    """
    if not _CUDA_KERNELS_AVAILABLE:
        # Fallback to Triton if CUDA kernels not available
        from vllm.attention.ops.triton_merge_attn_states import (
            merge_attn_states)
        return merge_attn_states(output, prefix_output, prefix_lse,
                                 suffix_output, suffix_lse, output_lse)

    num_tokens, num_heads, head_size = output.shape

    # Input validation
    assert prefix_output.shape == output.shape
    assert suffix_output.shape == output.shape
    assert prefix_lse.shape == (num_heads, num_tokens)
    assert suffix_lse.shape == (num_heads, num_tokens)

    if output_lse is not None:
        assert output_lse.shape == (num_heads, num_tokens)

    # Call optimized CUDA kernel
    ops.merge_attention_states(
        output,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        output_lse,
        num_tokens,
        num_heads,
        head_size,
    )


def merge_attn_states_optimized(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    """Optimized attention state merging with automatic backend selection.
    
    This function automatically selects between CUDA and Triton implementations
    based on availability and tensor characteristics for optimal performance.
    """
    # Use CUDA kernel for better performance when available
    if (_CUDA_KERNELS_AVAILABLE and output.device.type == 'cuda'
            and output.numel() > 1024):  # CUDA kernel efficiency
        merge_attn_states_cuda(output, prefix_output, prefix_lse,
                               suffix_output, suffix_lse, output_lse)
    else:
        # Fallback to Triton for smaller tensors or CUDA unavailable
        from vllm.attention.ops.triton_merge_attn_states import (
            merge_attn_states)
        merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                          suffix_lse, output_lse)


# Pure PyTorch fallback implementation for reference and testing
def merge_attn_states_torch(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    """Pure PyTorch implementation for reference and testing."""
    num_tokens, num_heads, head_size = output.shape

    # Handle inf values for FA2/FA3 compatibility
    prefix_lse = torch.where(prefix_lse == float('inf'),
                             torch.full_like(prefix_lse, float('-inf')),
                             prefix_lse)
    suffix_lse = torch.where(suffix_lse == float('inf'),
                             torch.full_like(suffix_lse, float('-inf')),
                             suffix_lse)

    # Compute max LSE for numerical stability
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    p_lse_norm = prefix_lse - max_lse
    s_lse_norm = suffix_lse - max_lse

    # Compute scale factors
    p_se = torch.exp(p_lse_norm)
    s_se = torch.exp(s_lse_norm)
    out_se = p_se + s_se

    # Store output LSE if requested
    if output_lse is not None:
        output_lse.copy_(torch.log(out_se) + max_lse)

    # Compute scaled outputs
    p_scale = (p_se / out_se).unsqueeze(-1)  # [num_heads, num_tokens, 1]
    s_scale = (s_se / out_se).unsqueeze(-1)  # [num_heads, num_tokens, 1]

    # Reshape for broadcasting: [num_tokens, num_heads, 1]
    p_scale = p_scale.transpose(0, 1)
    s_scale = s_scale.transpose(0, 1)

    # Merge attention outputs with proper scaling
    output.copy_(prefix_output * p_scale + suffix_output * s_scale)
