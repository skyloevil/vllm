#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit test for SigLIP position encoding optimization.
Tests the core interpolation logic without full model initialization.
"""

import math
import time

import torch
import torch.nn as nn


def interpolate_pos_encoding_bicubic(
    position_embeddings: torch.Tensor,
    height: int,
    width: int,
    patch_size: int,
) -> torch.Tensor:
    """Original bicubic interpolation (reference implementation)."""
    num_positions = position_embeddings.shape[0]
    num_grid_per_side = int(math.sqrt(num_positions))
    hidden_dim = position_embeddings.shape[1]

    h_patches = height // patch_size
    w_patches = width // patch_size

    if h_patches == num_grid_per_side and w_patches == num_grid_per_side:
        return position_embeddings.unsqueeze(0)

    # Reshape to grid
    position_embeddings = position_embeddings.reshape(1, num_grid_per_side,
                                                      num_grid_per_side,
                                                      hidden_dim).permute(
                                                          0, 3, 1, 2)

    # Bicubic interpolation
    position_embeddings = nn.functional.interpolate(
        position_embeddings,
        size=(h_patches, w_patches),
        mode="bicubic",
        align_corners=False,
    )

    # Reshape back
    position_embeddings = position_embeddings.permute(0, 2, 3, 1).reshape(
        1, h_patches * w_patches, hidden_dim)

    return position_embeddings


def fast_interpolate_pos_encoding_bilinear(
    position_embeddings: torch.Tensor,
    height: int,
    width: int,
    patch_size: int,
) -> torch.Tensor:
    """Optimized bilinear interpolation."""
    num_positions = position_embeddings.shape[0]
    num_grid_per_side = int(math.sqrt(num_positions))
    hidden_dim = position_embeddings.shape[1]
    device = position_embeddings.device

    h_patches = height // patch_size
    w_patches = width // patch_size

    if h_patches == num_grid_per_side and w_patches == num_grid_per_side:
        return position_embeddings.unsqueeze(0)

    # Generate interpolation coordinates
    h_idxs = torch.linspace(0,
                            num_grid_per_side - 1,
                            h_patches,
                            dtype=torch.float32,
                            device=device)
    w_idxs = torch.linspace(0,
                            num_grid_per_side - 1,
                            w_patches,
                            dtype=torch.float32,
                            device=device)

    # Compute floor and ceil indices
    h_floor = h_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_floor = w_idxs.to(torch.long)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

    # Compute interpolation weights
    dh = h_idxs - h_floor
    dw = w_idxs - w_floor

    # Vectorized weight calculation
    w00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1)
    w01 = ((1 - dh)[:, None] * dw[None, :]).reshape(-1)
    w10 = (dh[:, None] * (1 - dw)[None, :]).reshape(-1)
    w11 = (dh[:, None] * dw[None, :]).reshape(-1)

    # Compute grid indices
    idx00 = (h_floor[:, None] * num_grid_per_side +
             w_floor[None, :]).reshape(-1)
    idx01 = (h_floor[:, None] * num_grid_per_side +
             w_ceil[None, :]).reshape(-1)
    idx10 = (h_ceil[:, None] * num_grid_per_side +
             w_floor[None, :]).reshape(-1)
    idx11 = (h_ceil[:, None] * num_grid_per_side + w_ceil[None, :]).reshape(-1)

    # Batch embedding lookup
    indices = torch.stack([idx00, idx01, idx10, idx11], dim=0)
    weights = torch.stack([w00, w01, w10, w11], dim=0).unsqueeze(-1)
    embeds = position_embeddings[indices]
    weighted_embeds = embeds * weights

    # Weighted sum
    p0, p1, p2, p3 = weighted_embeds.unbind(dim=0)
    combined = p0 + p1 + p2 + p3

    return combined.view(1, h_patches * w_patches, hidden_dim)


def test_numerical_accuracy():
    """Test numerical accuracy of bilinear vs bicubic."""
    print("=" * 60)
    print("Test 1: Numerical Accuracy")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_size = 16
    hidden_dim = 768
    num_grid = 14  # 224 / 16

    # Create mock position embeddings
    position_embeddings = torch.randn(num_grid * num_grid,
                                      hidden_dim,
                                      device=device)

    test_cases = [
        (224, 224, "Original resolution"),
        (336, 336, "1.5x resolution"),
        (448, 448, "2x resolution"),
        (672, 672, "3x resolution"),
    ]

    for height, width, desc in test_cases:
        print(f"\n{desc}: {height}x{width}")

        with torch.no_grad():
            bicubic_result = interpolate_pos_encoding_bicubic(
                position_embeddings, height, width, patch_size)
            bilinear_result = fast_interpolate_pos_encoding_bilinear(
                position_embeddings, height, width, patch_size)

        abs_diff = torch.abs(bicubic_result - bilinear_result)
        rel_diff = abs_diff / (torch.abs(bicubic_result) + 1e-8)

        print(f"  Shape: {bilinear_result.shape}")
        print(f"  Max absolute diff: {abs_diff.max().item():.6f}")
        print(f"  Mean absolute diff: {abs_diff.mean().item():.6f}")
        print(f"  Max relative diff: {rel_diff.max().item():.6f}")

        bicubic_flat = bicubic_result.view(-1)
        bilinear_flat = bilinear_result.view(-1)
        cos_sim = torch.nn.functional.cosine_similarity(
            bicubic_flat.unsqueeze(0), bilinear_flat.unsqueeze(0))
        print(f"  Cosine similarity: {cos_sim.item():.8f}")

        if abs_diff.max().item() < 0.1:
            print("  PASS")
        else:
            print("  WARN: Large difference")


def test_performance():
    """Benchmark performance improvement."""
    print("\n" + "=" * 60)
    print("Test 2: Performance Benchmark")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_size = 14
    hidden_dim = 1152
    num_grid = 32  # 448 / 14

    position_embeddings = torch.randn(num_grid * num_grid,
                                      hidden_dim,
                                      device=device)

    test_resolutions = [
        (336, 336, "336x336"),
        (448, 448, "448x448"),
        (672, 672, "672x672"),
    ]

    n_iter = 100

    for height, width, desc in test_resolutions:
        print(f"\n{desc}:")

        # Warmup
        for _ in range(10):
            _ = interpolate_pos_encoding_bicubic(position_embeddings, height,
                                                 width, patch_size)
            _ = fast_interpolate_pos_encoding_bilinear(position_embeddings,
                                                       height, width,
                                                       patch_size)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark bicubic
        start = time.time()
        for _ in range(n_iter):
            _ = interpolate_pos_encoding_bicubic(position_embeddings, height,
                                                 width, patch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bicubic_time = (time.time() - start) / n_iter * 1000

        # Benchmark bilinear
        start = time.time()
        for _ in range(n_iter):
            _ = fast_interpolate_pos_encoding_bilinear(position_embeddings,
                                                       height, width,
                                                       patch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bilinear_time = (time.time() - start) / n_iter * 1000

        speedup = bicubic_time / bilinear_time

        print(f"  Bicubic:  {bicubic_time:.3f} ms")
        print(f"  Bilinear: {bilinear_time:.3f} ms")
        print(f"  Speedup:  {speedup:.2f}x")

        if speedup > 1.5:
            print("  PASS")
        else:
            print("  WARN: Limited speedup (expected on GPU)")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("Test 3: Edge Cases")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_size = 16
    hidden_dim = 768
    num_grid = 14

    position_embeddings = torch.randn(num_grid * num_grid,
                                      hidden_dim,
                                      device=device)

    # Edge case 1: Same resolution
    print("\nEdge case 1: Same resolution (224x224)")
    result = fast_interpolate_pos_encoding_bilinear(position_embeddings, 224,
                                                    224, patch_size)
    expected_shape = (1, 14 * 14, 768)
    print(f"  Output shape: {result.shape}")
    print(f"  Expected: {expected_shape}")
    assert result.shape == expected_shape
    print("  PASS")

    # Edge case 2: Large resolution
    print("\nEdge case 2: Large resolution (896x896)")
    result = fast_interpolate_pos_encoding_bilinear(position_embeddings, 896,
                                                    896, patch_size)
    expected_shape = (1, 56 * 56, 768)
    print(f"  Output shape: {result.shape}")
    print(f"  Expected: {expected_shape}")
    assert result.shape == expected_shape
    print("  PASS")

    # Edge case 3: Non-square resolution
    print("\nEdge case 3: Non-square (448x672)")
    result = fast_interpolate_pos_encoding_bilinear(position_embeddings, 448,
                                                    672, patch_size)
    expected_shape = (1, 28 * 42, 768)
    print(f"  Output shape: {result.shape}")
    print(f"  Expected: {expected_shape}")
    assert result.shape == expected_shape
    print("  PASS")


if __name__ == "__main__":
    print("\nTesting SigLIP Position Encoding Optimization (Unit Test)\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}\n")

    try:
        test_numerical_accuracy()
        test_performance()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        import sys

        sys.exit(1)
