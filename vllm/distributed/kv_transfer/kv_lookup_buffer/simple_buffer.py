# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
    Implements a distributed key-value (KV) cache transfer mechanism.

    Key Features:
    - Distributed KV cache transmission using PyNccl pipes.
    - Non-blocking `insert`, blocking `drop_select`.
    - Use CPU signal pipe to avoid racing condition
    - Handles buffer size constraints and provide backpressure mechanism to
      stop the prefill instance when the decode instance is slow.
"""
import threading
from collections import deque
from typing import Optional, Union, Dict, Tuple

import torch

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class SimpleBuffer(KVLookupBufferBase):

    def __init__(self, signal_pipe: KVPipeBase, data_pipe: KVPipeBase,
                 buffer_size_thresh: float):
        """
        signal_pipe: on CPU

        NOTE: on-device recv will block all threads in the process, making the
        KV cache producer unable to listen to new request while transmitting
        KV cache. Luckily CPU recv only blocks the current thread so we use
        CPU recv to listen to new request.

        data_pipe: on device (e.g. GPU)
        """

        self.buffer: deque[list[torch.Tensor]] = deque()
        # Hash table for O(1) lookup: hash_key -> buffer_index
        self.token_hash_table: Dict[str, int] = {}
        # Track buffer index for each item for efficient removal
        self.buffer_indices: deque[int] = deque()
        self._next_buffer_index = 0

        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_cv = threading.Condition()
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.request_handling_thread: Optional[threading.Thread] = None

        self.normal_signal = torch.tensor([0], device="cpu")
        self.end_signal = None

    def _compute_token_hash(self, tokens: torch.Tensor, roi: torch.Tensor) -> str:
        """
        Compute a hash key for tokens with ROI for O(1) lookup.
        
        Args:
            tokens: Input tokens tensor
            roi: Region of interest mask tensor
            
        Returns:
            String hash key for the token sequence
        """
        if tokens is None:
            # Empty request case - return special key
            return "*"
        
        # Extract relevant tokens using ROI mask
        relevant_tokens = tokens[roi]
        
        # Create hash from token values - convert to tuple for hashing
        # Use a combination of token values and length for better distribution
        token_tuple = tuple(relevant_tokens.cpu().numpy().tolist())
        hash_key = f"{len(token_tuple)}:{hash(token_tuple)}"
        
        return hash_key

    def _matches(self, tokens_roi_sender: list[torch.Tensor],
                 tokens_roi_recver: list[torch.Tensor]):

        # tokens_roi_sender: tokens and roi of the producer (in the buffer)
        # tokens_roi_recver: tokens and roi of the consumer (query)

        tokens_sender = tokens_roi_sender[0]
        tokens_recver = tokens_roi_recver[0]
        roi_sender = tokens_roi_sender[1]
        roi_recver = tokens_roi_recver[1]

        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

        # Assuming that roi is a binary mask on tokens
        tokens_sender = tokens_sender[roi_sender]
        tokens_recver = tokens_recver[roi_recver]

        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))
        if torch.allclose(tokens_sender[:min_length],
                          tokens_recver[:min_length]):
            return min_length

        return 0

    def _send_tensor_and_dec_size(self,
                                  tensor: Optional[torch.Tensor]) -> None:

        assert tensor is not None, "Use self.data_pipe.send(None) instead"
        self.buffer_size -= tensor.element_size() * tensor.numel()
        if tensor.dtype == torch.bool:
            tensor = tensor.float()
        self.data_pipe.send_tensor(tensor)

    def _get_element_size(self, data: Optional[Union[list, torch.Tensor]]):

        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if not data:
            # cannot perform `not data` on a tensor
            # so this check needs to go after the check above
            return 0

        raise AssertionError(f"Unknown data type {type(data)}")

    def _add_to_buffer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                       key: torch.Tensor, value: torch.Tensor,
                       hidden: torch.Tensor):

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()
        if isinstance(key, torch.Tensor):
            key = key.clone()
        if isinstance(value, torch.Tensor):
            value = value.clone()
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.clone()

        buffer_item = [input_tokens, roi, key, value, hidden]
        data_size = sum([self._get_element_size(data) for data in buffer_item])

        with self.buffer_cv:
            if self.buffer_size + data_size > self.buffer_size_threshold:
                # log outside the while loop to avoid this message being logged
                # repeatedly.
                logger.debug("KV transfer buffer is full. Handling...")
                while self.buffer_size + data_size > self.buffer_size_threshold:
                    self.buffer_cv.wait()

            # Compute hash key for the new item
            hash_key = self._compute_token_hash(input_tokens, roi)
            
            # Assign unique buffer index
            buffer_index = self._next_buffer_index
            self._next_buffer_index += 1
            
            self.buffer_size += data_size
            self.buffer.append(buffer_item)
            self.buffer_indices.append(buffer_index)
            
            # Add to hash table for O(1) lookup
            self.token_hash_table[hash_key] = buffer_index
            
            self.buffer_cv.notify()

    def _is_end_signal(self, signal):
        return signal is None

    def drop_select_handler(self):

        try:

            while True:
                signal = self.signal_pipe.recv_tensor()
                if self._is_end_signal(signal):
                    logger.info("Received end signal!")
                    break

                input_tokens = self.data_pipe.recv_tensor()

                roi = self.data_pipe.recv_tensor()
                assert roi is not None, "Please provide the roi when sending "\
                    "drop-select request"
                roi = (roi > 0.5)
                tokens_roi_recver = [input_tokens, roi]

                def is_buffer_available(
                    tokens_roi_recver: list[torch.Tensor], ) -> bool:
                    # Optimized O(1) hash table lookup instead of O(n) iteration
                    input_tokens = tokens_roi_recver[0]
                    roi = tokens_roi_recver[1]
                    
                    # Compute hash key for the query
                    query_hash_key = self._compute_token_hash(input_tokens, roi)
                    
                    # O(1) hash table lookup
                    if query_hash_key in self.token_hash_table:
                        return True
                        
                    # Special case: if query is empty ("*"), match any available item
                    if query_hash_key == "*" and len(self.buffer) > 0:
                        return True
                    
                    return False

                def find_and_remove_matching_item(
                    tokens_roi_recver: list[torch.Tensor], ) -> list[torch.Tensor]:
                    # Optimized O(1) lookup and removal
                    input_tokens = tokens_roi_recver[0]
                    roi = tokens_roi_recver[1]
                    
                    query_hash_key = self._compute_token_hash(input_tokens, roi)
                    
                    # Handle empty query case (drop select any item)
                    if query_hash_key == "*" and len(self.buffer) > 0:
                        # Remove first available item
                        matched_item = self.buffer.popleft()
                        removed_index = self.buffer_indices.popleft()
                        
                        # Clean up hash table - find and remove the corresponding entry
                        for key, index in list(self.token_hash_table.items()):
                            if index == removed_index:
                                del self.token_hash_table[key]
                                break
                        
                        return matched_item
                    
                    # Normal case: find exact match using hash table
                    if query_hash_key in self.token_hash_table:
                        target_index = self.token_hash_table[query_hash_key]
                        
                        # Find the item in buffer with matching index
                        for i, (item, item_index) in enumerate(zip(self.buffer, self.buffer_indices)):
                            if item_index == target_index:
                                # Remove item from buffer and indices
                                self.buffer.remove(item)
                                self.buffer_indices.remove(item_index)
                                
                                # Remove from hash table
                                del self.token_hash_table[query_hash_key]
                                
                                return item
                    
                    # Should not reach here if is_buffer_available returned True
                    raise RuntimeError("Matching item not found despite availability check")

                with self.buffer_cv:
                    while not is_buffer_available(tokens_roi_recver):
                        logger.debug(
                            "KV transfer buffer is not available. Waiting...")
                        self.buffer_cv.wait()
                    # need to clone the tensor
                    # in case the tensor is freed before sending finishes
                    matched_item = find_and_remove_matching_item(tokens_roi_recver)
                    for tensor in matched_item:
                        self._send_tensor_and_dec_size(tensor)
                    self.buffer_cv.notify()

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")

    def drop_select(
            self, input_tokens: Optional[torch.Tensor],
            roi: Optional[torch.Tensor]) -> list[Optional[torch.Tensor]]:

        assert self.request_handling_thread is None, \
            "drop_select should be called by the KV cache consumer "\
            "(e.g. the decode vLLM instance)"

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone().float()

        self.signal_pipe.send_tensor(self.normal_signal)
        self.data_pipe.send_tensor(input_tokens)
        self.data_pipe.send_tensor(roi)

        input_tokens = self.data_pipe.recv_tensor()
        roi = self.data_pipe.recv_tensor()
        if roi is not None:
            # convert from float tensor to bool tensor
            # as PyNccl does not support sending bool tensor
            roi = (roi > 0.5)
        key = self.data_pipe.recv_tensor()
        value = self.data_pipe.recv_tensor()
        hidden = self.data_pipe.recv_tensor()

        return [input_tokens, roi, key, value, hidden]

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:

        self._add_to_buffer(input_tokens, roi, key, value, hidden)

        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread is None:
            self.request_handling_thread = threading.Thread(
                target=self.drop_select_handler)
            self.request_handling_thread.start()

    def close(self):

        if hasattr(self, "request_handling_thread"
                   ) and self.request_handling_thread is not None:
            self.request_handling_thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)
