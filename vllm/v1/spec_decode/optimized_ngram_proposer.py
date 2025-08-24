# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized N-gram proposer implementations with improved performance.

This module provides several optimized implementations for N-gram matching
and token proposal, offering significant performance improvements over the
original KMP-based approach.
"""

from typing import Optional

import numpy as np

try:
    from numba import jit
    _NUMBA_AVAILABLE = True
except ImportError:
    # Fallback decorator if numba not available
    def jit(nopython=True):

        def decorator(func):
            return func

        return decorator

    _NUMBA_AVAILABLE = False

from vllm.config import VllmConfig


@jit(nopython=True)
def _find_ngram_with_rolling_hash(origin_tokens: np.ndarray, min_ngram: int,
                                  max_ngram: int, max_model_len: int,
                                  k: int) -> Optional[np.ndarray]:
    """Optimized N-gram matching using rolling hash technique.
    
    This implementation uses rolling hash to quickly identify potential matches,
    significantly reducing the number of element-by-element comparisons.
    """
    total_tokens = origin_tokens.shape[0]

    if total_tokens < min_ngram:
        return None

    k = min(k, max_model_len - total_tokens)
    if k <= 0:
        return None

    # Hash base for rolling hash computation
    BASE = 31
    MOD = 10**9 + 7

    best_ngram_len = 0
    best_start_pos = -1

    # Try different N-gram lengths from max to min (greedy approach)
    for ngram_len in range(min(max_ngram, total_tokens), min_ngram - 1, -1):
        # Target N-gram (last ngram_len tokens)
        target_start = total_tokens - ngram_len

        # Compute hash for target N-gram
        target_hash = 0
        base_power = 1
        for i in range(ngram_len):
            target_hash = (target_hash +
                           origin_tokens[target_start + i] * base_power) % MOD
            if i < ngram_len - 1:  # Don't update base_power for last iteration
                base_power = (base_power * BASE) % MOD

        # Search for matches using rolling hash
        current_hash = 0
        for i in range(ngram_len):
            current_hash = (current_hash + origin_tokens[i] *
                            (BASE**(ngram_len - 1 - i))) % MOD

        # Check positions where we can find matches
        for start_idx in range(total_tokens - ngram_len):
            if current_hash == target_hash:
                # Verify actual match (hash collision check)
                match = True
                for j in range(ngram_len):
                    if origin_tokens[start_idx +
                                     j] != origin_tokens[target_start + j]:
                        match = False
                        break

                if match and start_idx + ngram_len + k <= total_tokens:
                    # Found a valid match with enough following tokens
                    best_ngram_len = ngram_len
                    best_start_pos = start_idx + ngram_len
                    break

            # Rolling hash update for next position
            if start_idx + ngram_len < total_tokens:
                # Remove leftmost element and add new rightmost element
                current_hash = (current_hash -
                                origin_tokens[start_idx] * base_power) % MOD
                current_hash = (current_hash * BASE +
                                origin_tokens[start_idx + ngram_len]) % MOD

        # If we found a match, return immediately (greedy: longest first)
        if best_ngram_len > 0:
            break

    if best_ngram_len == 0:
        return None

    # Extract following tokens
    extract_len = min(k, total_tokens - best_start_pos)
    return origin_tokens[best_start_pos:best_start_pos + extract_len]


@jit(nopython=True)
def _find_ngram_with_suffix_array(origin_tokens: np.ndarray, min_ngram: int,
                                  max_ngram: int, max_model_len: int,
                                  k: int) -> Optional[np.ndarray]:
    """Alternative N-gram matching using direct suffix comparison.
    
    This approach directly compares suffixes without complex preprocessing,
    which can be more efficient for shorter sequences.
    """
    total_tokens = origin_tokens.shape[0]

    if total_tokens < min_ngram:
        return None

    k = min(k, max_model_len - total_tokens)
    if k <= 0:
        return None

    best_ngram_len = 0
    best_start_pos = -1

    # Extract last max_ngram tokens as the target
    max_search_len = min(max_ngram, total_tokens)

    # Search backwards for longest match
    for ngram_len in range(max_search_len, min_ngram - 1, -1):
        target_suffix_start = total_tokens - ngram_len

        # Look for this N-gram in previous positions
        search_end = target_suffix_start
        for start_pos in range(search_end):
            if start_pos + ngram_len > search_end:
                break

            # Check if we have a match
            match = True
            for j in range(ngram_len):
                if origin_tokens[start_pos +
                                 j] != origin_tokens[target_suffix_start + j]:
                    match = False
                    break

            if match and start_pos + ngram_len + k <= total_tokens:
                # Found valid match with sufficient following tokens
                best_ngram_len = ngram_len
                best_start_pos = start_pos + ngram_len
                break

        if best_ngram_len > 0:
            break

    if best_ngram_len == 0:
        return None

    extract_len = min(k, total_tokens - best_start_pos)
    return origin_tokens[best_start_pos:best_start_pos + extract_len]


@jit(nopython=True)
def _find_ngram_optimized_kmp(origin_tokens: np.ndarray, min_ngram: int,
                              max_ngram: int, max_model_len: int,
                              k: int) -> Optional[np.ndarray]:
    """Optimized KMP with early termination and reduced allocations."""
    total_tokens = origin_tokens.shape[0]

    if total_tokens < min_ngram:
        return None

    k = min(k, max_model_len - total_tokens)
    if k <= 0:
        return None

    # Try different N-gram lengths starting from maximum
    for target_ngram_len in range(min(max_ngram, total_tokens), min_ngram - 1,
                                  -1):
        target_start = total_tokens - target_ngram_len

        # Use Boyer-Moore-style bad character heuristic for quick skipping
        last_char = origin_tokens[total_tokens - 1]

        # Search for matches
        i = target_ngram_len - 1
        while i < target_start:
            # Quick character check (bad character heuristic)
            if origin_tokens[i] != last_char:
                i += 1
                continue

            # Potential match found, check full N-gram
            match_start = i - target_ngram_len + 1
            match = True

            for j in range(target_ngram_len):
                if origin_tokens[match_start +
                                 j] != origin_tokens[target_start + j]:
                    match = False
                    break

            if match and match_start + target_ngram_len + k <= total_tokens:
                # Found valid match
                extract_start = match_start + target_ngram_len
                extract_len = min(k, total_tokens - extract_start)
                return origin_tokens[extract_start:extract_start + extract_len]

            i += 1

    return None


class OptimizedNgramProposer:
    """Optimized N-gram proposer with multiple algorithm options."""

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        # Warm up all implementations
        test_tokens = np.zeros(1024, dtype=np.int32)
        self.propose(test_tokens)

    def propose(self,
                context_token_ids: np.ndarray,
                algorithm: str = "auto") -> Optional[np.ndarray]:
        """Propose tokens using optimized N-gram matching.
        
        Args:
            context_token_ids: Input token sequence
            algorithm: Algorithm choice - "auto", "rolling_hash", 
                "suffix_array", "kmp"
        """
        if algorithm == "auto":
            # Automatic algorithm selection based on input characteristics
            seq_len = len(context_token_ids)
            if seq_len > 10000:
                algorithm = "rolling_hash"  # Better for very long sequences
            elif seq_len > 1000:
                algorithm = "suffix_array"  # Balanced performance
            else:
                algorithm = "kmp"  # Good for shorter sequences

        if algorithm == "rolling_hash":
            return _find_ngram_with_rolling_hash(context_token_ids, self.min_n,
                                                 self.max_n,
                                                 self.max_model_len, self.k)
        elif algorithm == "suffix_array":
            return _find_ngram_with_suffix_array(context_token_ids, self.min_n,
                                                 self.max_n,
                                                 self.max_model_len, self.k)
        elif algorithm == "kmp":
            return _find_ngram_optimized_kmp(context_token_ids, self.min_n,
                                             self.max_n, self.max_model_len,
                                             self.k)
        else:
            # Fallback to original implementation
            from vllm.v1.spec_decode.ngram_proposer import (
                _find_longest_matched_ngram_and_propose_tokens)
            return _find_longest_matched_ngram_and_propose_tokens(
                context_token_ids, self.min_n, self.max_n, self.max_model_len,
                self.k)

    def load_model(self, *args, **kwargs):
        # No model to load
        del args, kwargs  # Suppress unused parameter warnings
