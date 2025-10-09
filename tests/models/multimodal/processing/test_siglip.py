#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test script for SigLIP position encoding interpolation optimization.

This script validates:
1. Numerical accuracy (bilinear vs bicubic)
2. Performance improvement
3. Edge cases
"""

# Import the optimized SigLIP implementation
import sys
import time

import torch
from transformers import SiglipVisionConfig

from vllm.distributed import initialize_model_parallel
from vllm.model_executor.models.siglip import SiglipVisionEmbeddings


def test_numerical_accuracy():
    """Check that bilinear interpolation matches bicubic outputs."""
    print("=" * 60)
    print("Test 1: Numerical Accuracy")
    print("=" * 60)

    # Create config
    config = SiglipVisionConfig(
        hidden_size=768,
        image_size=224,  # 14x14 patches
        patch_size=16,
        num_channels=3,
        num_attention_heads=12,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_layer = SiglipVisionEmbeddings(config).to(device)

    # Test different resolutions
    test_cases = [
        (224, 224, "Original resolution (no interpolation)"),
        (336, 336, "1.5x resolution"),
        (448, 448, "2x resolution"),
        (672, 672, "3x resolution"),
    ]

    for height, width, desc in test_cases:
        print(f"\n{desc}: {height}x{width}")

        # Create dummy input
        pixel_values = torch.randn(1, 3, height, width).to(device)
        patch_embeds = embeddings_layer.patch_embedding(pixel_values)
        test_embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # Compare outputs
        with torch.no_grad():
            bicubic_result = embeddings_layer.interpolate_pos_encoding(
                test_embeddings, height, width)
            bilinear_result = embeddings_layer.fast_interpolate_pos_encoding(
                test_embeddings, height, width)

        # Calculate differences
        abs_diff = torch.abs(bicubic_result - bilinear_result)
        rel_diff = abs_diff / (torch.abs(bicubic_result) + 1e-8)

        print(f"  Shape: {bilinear_result.shape}")
        print(f"  Max absolute diff: {abs_diff.max().item():.6f}")
        print(f"  Mean absolute diff: {abs_diff.mean().item():.6f}")
        print(f"  Max relative diff: {rel_diff.max().item():.6f}")
        print(f"  Mean relative diff: {rel_diff.mean().item():.6f}")

        # Cosine similarity
        bicubic_flat = bicubic_result.view(-1)
        bilinear_flat = bilinear_result.view(-1)
        cos_sim = torch.nn.functional.cosine_similarity(
            bicubic_flat.unsqueeze(0), bilinear_flat.unsqueeze(0))
        print(f"  Cosine similarity: {cos_sim.item():.8f}")

        # Check if differences are acceptable
        if abs_diff.max().item() < 0.1:
            print("  PASS: Max diff < 0.1")
        else:
            print("  WARN: Max diff >= 0.1")


def test_performance():
    """Benchmark the performance improvement."""
    print("\n" + "=" * 60)
    print("Test 2: Performance Benchmark")
    print("=" * 60)

    config = SiglipVisionConfig(
        hidden_size=1024,  # Larger hidden size for more realistic test
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_attention_heads=16,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_layer = SiglipVisionEmbeddings(config).to(device)

    # Test resolutions
    test_resolutions = [
        (336, 336, "336x336"),
        (448, 448, "448x448"),
        (672, 672, "672x672"),
    ]

    n_iter = 100

    for height, width, desc in test_resolutions:
        print(f"\n{desc}:")

        pixel_values = torch.randn(1, 3, height, width).to(device)
        patch_embeds = embeddings_layer.patch_embedding(pixel_values)
        test_embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # Warmup
        for _ in range(10):
            _ = embeddings_layer.interpolate_pos_encoding(
                test_embeddings, height, width)
            _ = embeddings_layer.fast_interpolate_pos_encoding(
                test_embeddings, height, width)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark bicubic
        start = time.time()
        for _ in range(n_iter):
            _ = embeddings_layer.interpolate_pos_encoding(
                test_embeddings, height, width)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bicubic_time = (time.time() - start) / n_iter * 1000

        # Benchmark bilinear
        start = time.time()
        for _ in range(n_iter):
            _ = embeddings_layer.fast_interpolate_pos_encoding(
                test_embeddings, height, width)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bilinear_time = (time.time() - start) / n_iter * 1000

        speedup = bicubic_time / bilinear_time

        print(f"  Bicubic:  {bicubic_time:.3f} ms")
        print(f"  Bilinear: {bilinear_time:.3f} ms")
        print(f"  Speedup:  {speedup:.2f}x")

        if speedup > 1.5:
            print("  PASS: Speedup > 1.5x")
        else:
            print("  WARN: Speedup < 1.5x")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("Test 3: Edge Cases")
    print("=" * 60)

    config = SiglipVisionConfig(
        hidden_size=768,
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_attention_heads=12,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_layer = SiglipVisionEmbeddings(config).to(device)

    # Edge case 1: Same resolution (no interpolation needed)
    print("\nEdge case 1: Same resolution (224x224)")
    pixel_values = torch.randn(1, 3, 224, 224).to(device)
    patch_embeds = embeddings_layer.patch_embedding(pixel_values)
    test_embeddings = patch_embeds.flatten(2).transpose(1, 2)

    result = embeddings_layer.fast_interpolate_pos_encoding(
        test_embeddings, 224, 224)
    expected_shape = (1, 14 * 14, 768)
    print(f"  Output shape: {result.shape}")
    print(f"  Expected shape: {expected_shape}")
    assert result.shape == expected_shape, "Shape mismatch!"
    print("  PASS")

    # Edge case 2: Very large resolution
    print("\nEdge case 2: Large resolution (896x896)")
    pixel_values = torch.randn(1, 3, 896, 896).to(device)
    patch_embeds = embeddings_layer.patch_embedding(pixel_values)
    test_embeddings = patch_embeds.flatten(2).transpose(1, 2)

    result = embeddings_layer.fast_interpolate_pos_encoding(
        test_embeddings, 896, 896)
    expected_shape = (1, 56 * 56, 768)
    print(f"  Output shape: {result.shape}")
    print(f"  Expected shape: {expected_shape}")
    assert result.shape == expected_shape, "Shape mismatch!"
    print("  PASS")

    # Edge case 3: Non-square resolution
    print("\nEdge case 3: Non-square resolution (448x672)")
    pixel_values = torch.randn(1, 3, 448, 672).to(device)
    patch_embeds = embeddings_layer.patch_embedding(pixel_values)
    test_embeddings = patch_embeds.flatten(2).transpose(1, 2)

    result = embeddings_layer.fast_interpolate_pos_encoding(
        test_embeddings, 448, 672)
    expected_shape = (1, 28 * 42, 768)
    print(f"  Output shape: {result.shape}")
    print(f"  Expected shape: {expected_shape}")
    assert result.shape == expected_shape, "Shape mismatch!"
    print("  PASS")


def test_gradient_flow():
    """Ensure gradients flow through the optimized interpolation."""
    print("\n" + "=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)

    config = SiglipVisionConfig(
        hidden_size=768,
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_attention_heads=12,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_layer = SiglipVisionEmbeddings(config).to(device)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pixel_values = torch.randn(1, 3, 448, 448, requires_grad=True).to(device)
    patch_embeds = embeddings_layer.patch_embedding(pixel_values)
    test_embeddings = patch_embeds.flatten(2).transpose(1, 2)

    # Forward pass
    result = embeddings_layer.fast_interpolate_pos_encoding(
        test_embeddings, 448, 448)

    # Backward pass
    loss = result.sum()
    loss.backward()

    print(f"  Result shape: {result.shape}")
    print(f"  Gradient shape: {pixel_values.grad.shape}")
    print(f"  Gradient norm: {pixel_values.grad.norm().item():.6f}")

    if pixel_values.grad is not None and pixel_values.grad.norm() > 0:
        print("  PASS: Gradients flow correctly")
    else:
        print("  FAIL: No gradients!")


if __name__ == "__main__":
    print("\n Testing SigLIP Position Encoding Optimization\n")

    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (slower but functional)")
    print(f"PyTorch version: {torch.__version__}\n")

    # Initialize vLLM distributed environment
    # Required for VocabParallelEmbedding used in position_embedding
    print("Initializing model parallel (tensor_parallel_size=1)...")
    initialize_model_parallel(tensor_model_parallel_size=1)
    print("Initialization complete.\n")

    try:
        test_numerical_accuracy()
        test_performance()
        test_edge_cases()
        test_gradient_flow()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
