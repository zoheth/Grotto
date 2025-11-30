"""
Ring Buffer-based KV Cache for CUDA Graph compatibility.

This implementation uses a circular buffer with GPU-tensor based indexing
to eliminate CPU control flow and dynamic memory operations, making it
fully compatible with CUDA Graph capture and replay.

Key differences from ActionCache:
1. All indices are GPU tensors (no CPU scalars)
2. Uses modulo arithmetic for circular buffer (no memory shifting)
3. Returns fixed-shape tensors + attention mask (no dynamic shapes)
4. No CPU-GPU synchronization (.item() calls) in the forward path
"""

import torch
from typing import Tuple, Optional


class RingBufferActionCache:
    """
    CUDA Graph-compatible KV Cache using Ring Buffer mechanism.

    This cache maintains a circular buffer of KV pairs and uses index_select
    to reconstruct time-ordered windows for attention computation.

    Design principles:
    - All state is stored in GPU tensors
    - All operations are deterministic and graph-capturable
    - Output shapes are fixed (determined by max_attention_size)
    - Attention masking handles variable-length sequences

    Args:
        batch_size: Batch size (can be B*S for spatial batching)
        max_seq_len: Maximum sequence length (ring buffer capacity)
        num_heads: Number of attention heads
        head_dim: Dimension per head
        device: Device to place tensors
        dtype: Data type for cache tensors
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Pre-allocate KV Cache (circular buffer)
        self.k_cache = torch.zeros(
            (batch_size, max_seq_len, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (batch_size, max_seq_len, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # State as GPU Tensors (Critical for CUDA Graph compatibility)
        # Current write position in the ring buffer [0, max_seq_len-1]
        self.pos_ptr = torch.zeros(1, dtype=torch.long, device=device)

        # Total number of tokens processed (for tracking cache fill state)
        self.total_tokens = torch.zeros(1, dtype=torch.long, device=device)

        # Pre-compute index buffer for efficient gathering
        # This avoids allocating new tensors in the hot path
        self._arange = torch.arange(max_seq_len, device=device, dtype=torch.long)

        # Pre-allocate mask tensors (for output)
        self._attention_mask_buffer = torch.zeros(
            max_seq_len, dtype=torch.bool, device=device
        )

        # Pre-allocate scalar tensors for CUDA Graph compatibility
        # These avoid creating new tensors during graph capture
        self._window_len_tensor = torch.zeros(1, dtype=torch.long, device=device)

    def reset(self):
        """
        Reset cache to empty state.

        Note: If CUDA Graph is already captured, calling this may invalidate
        the graph. Reset should typically be done outside of captured regions.
        """
        self.pos_ptr.zero_()
        self.total_tokens.zero_()

    def update_and_get_window(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        num_new_tokens: int = 1,
        max_attention_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update cache with new KV pairs and return attention window + mask.

        This method is designed for CUDA Graph compatibility:
        - All operations are deterministic
        - No CPU control flow based on tensor values
        - Output shapes are fixed
        - Returns attention mask to handle variable lengths

        Args:
            k: New keys [BS, num_new_tokens, num_heads, head_dim]
            v: New values [BS, num_new_tokens, num_heads, head_dim]
            num_new_tokens: Number of new tokens (should be constant for Graph capture)
            max_attention_size: Maximum attention window size (default: max_seq_len)

        Returns:
            k_window: Keys in time-ordered window [BS, window_len, num_heads, head_dim]
            v_window: Values in time-ordered window [BS, window_len, num_heads, head_dim]
            attention_mask: Valid token mask [window_len] (True = valid, False = padding)
        """
        if max_attention_size is None:
            max_attention_size = self.max_seq_len

        # Ensure window size doesn't exceed cache capacity
        window_len = min(max_attention_size, self.max_seq_len)

        # --- Phase 1: Write new tokens to ring buffer (O(num_new_tokens)) ---

        # Calculate write indices using modulo arithmetic
        # Example: if pos_ptr=1022, max_seq_len=1024, num_new_tokens=5
        # write_indices = [1022, 1023, 0, 1, 2] (wraps around)
        write_indices = (
            self.pos_ptr + torch.arange(num_new_tokens, device=self.device)
        ) % self.max_seq_len

        # Scatter write to ring buffer (in-place update)
        # This is much faster than shifting the entire cache
        self.k_cache[:, write_indices] = k
        self.v_cache[:, write_indices] = v

        # Update state pointers (in-place, no allocation)
        self.pos_ptr.add_(num_new_tokens).fmod_(self.max_seq_len)
        self.total_tokens.add_(num_new_tokens)

        # --- Phase 2: Read window from ring buffer (O(window_len)) ---

        # Compute read indices for the attention window
        # We want the last 'window_len' tokens in time order
        # Formula: (current_pos - window_len, ..., current_pos - 1) % max_seq_len
        #
        # Example: pos_ptr=5, window_len=8, max_seq_len=1024
        # Logical positions: [-3, -2, -1, 0, 1, 2, 3, 4]
        # Physical indices: [1021, 1022, 1023, 0, 1, 2, 3, 4]
        read_indices = (
            self.pos_ptr - window_len + self._arange[:window_len]
        ) % self.max_seq_len

        # Gather data from ring buffer to create time-ordered window
        # This is a memory copy but much more predictable than dynamic slicing
        k_window = self.k_cache.index_select(1, read_indices)
        v_window = self.v_cache.index_select(1, read_indices)

        # --- Phase 3: Create attention mask for valid tokens ---

        # Compute how many tokens are actually valid (min of total_tokens and window_len)
        # This handles the case where cache is not yet full
        #
        # Example cases:
        # - If total_tokens=3, window_len=8 -> mask=[1,1,1,0,0,0,0,0]
        # - If total_tokens=1000, window_len=8 -> mask=[1,1,1,1,1,1,1,1]

        # Create mask: positions < valid_len are True (valid)
        # Note: We use torch operations to avoid .item() which breaks CUDA Graph
        # Also avoid torch.tensor() during graph capture by using pre-allocated buffer
        self._window_len_tensor.fill_(window_len)
        valid_len = torch.minimum(self.total_tokens, self._window_len_tensor)

        # Create attention mask (True = valid token, False = padding)
        # Mask shape: [window_len]
        attention_mask = self._arange[:window_len] >= (window_len - valid_len)

        return k_window, v_window, attention_mask

    def get_state_dict(self) -> dict:
        """
        Get current cache state for checkpointing.

        Returns:
            Dictionary containing cache state
        """
        return {
            'k_cache': self.k_cache.clone(),
            'v_cache': self.v_cache.clone(),
            'pos_ptr': self.pos_ptr.clone(),
            'total_tokens': self.total_tokens.clone(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load cache state from checkpoint.

        Args:
            state_dict: Dictionary containing cache state
        """
        self.k_cache.copy_(state_dict['k_cache'])
        self.v_cache.copy_(state_dict['v_cache'])
        self.pos_ptr.copy_(state_dict['pos_ptr'])
        self.total_tokens.copy_(state_dict['total_tokens'])