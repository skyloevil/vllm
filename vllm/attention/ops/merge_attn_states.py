# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:

    # Enhanced dtype support for better performance coverage
    # Added experimental FP8 support for modern quantized models
    def supported_dtypes(o: torch.Tensor) -> bool:
        supported = [torch.float32, torch.half, torch.bfloat16]
        # Add FP8 types if available (experimental optimization)
        if hasattr(torch, 'float8_e4m3fn'):
            supported.extend([torch.float8_e4m3fn, torch.float8_e5m2])
        return o.dtype in supported

    # NOTE(DefTruth): headdim must be multiple of pack_size for CUDA kernel
    # pack_size = 16 / sizeof(dtype), so dtype alignment is required
    def supported_headdim(o: torch.Tensor) -> bool:
        headdim = o.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        if o.dtype == torch.float32:
            pack_size = 4  # 16 / 4
            return headdim % pack_size == 0
        else:  # half, bfloat16
            pack_size = 8  # 16 / 2
            return headdim % pack_size == 0

    # Performance optimization: prioritize CUDA when conditions are met
    use_cuda = (current_platform.is_cuda() and supported_dtypes(output)
                and supported_headdim(output))

    if use_cuda:
        try:
            from vllm._custom_ops import merge_attn_states
            logger.debug(
                f"Using CUDA merge_attn_states for shape {output.shape}, dtype {output.dtype}"
            )
            return merge_attn_states(output, prefix_output, prefix_lse,
                                     suffix_output, suffix_lse, output_lse)
        except Exception as e:
            logger.warning(
                f"CUDA merge_attn_states failed, fallback to Triton: {e}")
    else:
        # Log fallback reasons for performance analysis
        reasons = []
        if not current_platform.is_cuda():
            reasons.append("non-CUDA platform")
        if not supported_dtypes(output):
            reasons.append(f"unsupported dtype {output.dtype}")
        if not supported_headdim(output):
            reasons.append(f"unsupported headdim {output.shape[2]}")
        logger.debug(
            f"Fallback to Triton merge_attn_states: {', '.join(reasons)}")

    # Fallback to Triton implementation
    from vllm.attention.ops.triton_merge_attn_states import merge_attn_states
    return merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                             suffix_lse, output_lse)
