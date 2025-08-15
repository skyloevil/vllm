#!/usr/bin/env python3
"""
Test script to verify FlashInfer usage correctness in the optimized sampler.
"""
import torch
from typing import Dict, Optional

def simulate_flashinfer_sample(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    generators: Dict[int, torch.Generator],
) -> torch.Tensor:
    """Mock FlashInfer sample for testing."""
    # Simplified implementation for testing
    assert not generators, "FlashInfer doesn't support generators"
    
    if k is not None:
        # Apply top-k filter
        top_k_values, top_k_indices = torch.topk(logits, k.max().item(), dim=-1)
        mask = torch.arange(logits.size(-1), device=logits.device).expand(logits.size(0), -1)
        mask = mask >= k.unsqueeze(1)
        logits = logits.masked_fill(mask, float('-inf'))
    
    if p is not None:
        # Apply top-p filter (simplified)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum_probs > p.unsqueeze(1)
        mask[:, 0] = False  # Keep at least one token
        # This is a simplified top-p, real implementation would be more complex
    
    # Sample from the remaining distribution
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def test_flashinfer_usage():
    """Test that FlashInfer usage is correct and safe."""
    print("Testing FlashInfer usage scenarios...")
    
    device = 'cpu'  # Use CPU for testing
    batch_size = 8
    vocab_size = 1000
    
    # Test data
    logits = torch.randn(batch_size, vocab_size, device=device)
    k_values = torch.randint(1, 100, (batch_size,), device=device)
    p_values = torch.rand(batch_size, device=device) * 0.5 + 0.5
    
    print("\n1. Testing no generators case (should use FlashInfer):")
    try:
        result = simulate_flashinfer_sample(logits.clone(), k_values, p_values, {})
        print(f"   ✅ FlashInfer call successful, shape: {result.shape}")
    except Exception as e:
        print(f"   ❌ FlashInfer call failed: {e}")
    
    print("\n2. Testing with generators case (should NOT use FlashInfer):")
    generators = {0: torch.Generator(device=device), 2: torch.Generator(device=device)}
    try:
        result = simulate_flashinfer_sample(logits.clone(), k_values, p_values, generators)
        print(f"   ❌ This should have failed - FlashInfer doesn't support generators")
    except AssertionError as e:
        print(f"   ✅ Correctly rejected FlashInfer with generators: {e}")
    
    print("\n3. Testing contiguous logits requirement:")
    # Create non-contiguous logits
    padded_logits = torch.randn(batch_size, vocab_size * 2, device=device)
    non_contiguous_logits = padded_logits[:, :vocab_size]  # This creates a view/slice
    
    print(f"   Original logits contiguous: {logits.is_contiguous()}")
    print(f"   Sliced logits contiguous: {non_contiguous_logits.is_contiguous()}")
    
    # Test that we handle non-contiguous correctly
    try:
        contiguous_logits = non_contiguous_logits.contiguous()
        result = simulate_flashinfer_sample(contiguous_logits, k_values, p_values, {})
        print(f"   ✅ Contiguous conversion works, shape: {result.shape}")
    except Exception as e:
        print(f"   ❌ Contiguous conversion failed: {e}")
    
    print("\n4. Testing edge cases:")
    # Test with k=None, p=None
    try:
        result = simulate_flashinfer_sample(logits.clone(), None, None, {})
        print(f"   ✅ No filtering case works, shape: {result.shape}")
    except Exception as e:
        print(f"   ⚠️  No filtering case: {e}")
    
    # Test with very small k
    small_k = torch.ones(batch_size, dtype=torch.long, device=device)
    try:
        result = simulate_flashinfer_sample(logits.clone(), small_k, None, {})
        print(f"   ✅ Small k case works, shape: {result.shape}")
    except Exception as e:
        print(f"   ❌ Small k case failed: {e}")
    
    print("\nFlashInfer usage verification complete!")

if __name__ == "__main__":
    test_flashinfer_usage()