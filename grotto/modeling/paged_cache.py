import math
import torch
from typing import Optional, Tuple

from flashinfer import BatchPrefillWithPagedKVCacheWrapper

class PagedCache:
    def __init__(
            self,
            max_total_tokens: int,
            page_size: int,
            num_heads: int,
            head_dim: int,
            sink_size: int = 0,
            dtype: torch.dtype = torch.bfloat16,
            device: torch.device = "cuda" # type: ignore
    ):
        self.page_size = page_size
        self.sink_size = sink_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Calculate number of pages needed
        self.max_pages = math.ceil(max_total_tokens / page_size)

        # Pre-allocate cache tensors
        # Shape: [max_pages, page_size, num_heads, head_dim]
        self.k_cache = torch.zeros(
            (self.max_pages, page_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.zeros(
            (self.max_pages, page_size, num_heads, head_dim),
            dtype=dtype,
            device=device
        )

        # Page management
        self.next_free_page_id = 0
        self.active_page_indices: list[int] = []
        self.free_page_pool: list[int] = []  # Recycled pages available for reuse

        # Position tracking within current page
        self.current_page_offset = 0
        self.seq_len = 0

        # Track global position (for RoPE compatibility and denoising overwrite detection)
        self.global_position = 0
        # Track the global end index (like dict cache's global_end_index)
        # This is used to detect if we're overwriting the same position during denoising
        self.global_end_index = 0

    def reset(self):
        """Reset the cache to initial state without reallocating memory."""
        self.next_free_page_id = 0
        self.active_page_indices = []
        self.free_page_pool = []
        self.current_page_offset = 0
        self.seq_len = 0
        self.global_position = 0
        self.global_end_index = 0

    def _allocate_page(self) -> int:
        if self.free_page_pool:
            return self.free_page_pool.pop()
        
        # Allocate new page
        if self.next_free_page_id >= self.max_pages:
            raise RuntimeError(
                f"KV Cache Out of Memory! "
                f"Tried to allocate page {self.next_free_page_id} but max is {self.max_pages}. "
                f"Current seq_len: {self.seq_len}, max_tokens: {self.max_pages * self.page_size}. "
                f"Call evict() before append() to free space."
            )

        page_id = self.next_free_page_id
        self.next_free_page_id += 1
        return page_id


    def append(self, k: torch.Tensor, v: torch.Tensor):
        
        incoming_len = k.shape[0]

        incoming_processed = 0
        while incoming_processed < incoming_len:
            # Allocate new page if needed
            if not self.active_page_indices or self.current_page_offset == self.page_size:
                new_page_id = self._allocate_page()
                self.active_page_indices.append(new_page_id)
                self.current_page_offset = 0

            # Calculate how much to write to current page
            current_page_id = self.active_page_indices[-1]
            space_left = self.page_size - self.current_page_offset
            to_write = min(space_left, incoming_len - incoming_processed)

            # Write to cache
            self.k_cache[
                current_page_id,
                self.current_page_offset:self.current_page_offset + to_write
            ] = k[incoming_processed:incoming_processed + to_write]

            self.v_cache[
                current_page_id,
                self.current_page_offset:self.current_page_offset + to_write
            ] = v[incoming_processed:incoming_processed + to_write]

            self.current_page_offset += to_write
            incoming_processed += to_write

        # Update seq_len after successful append
        self.seq_len += incoming_len
        self.global_position += incoming_len

    def update_or_append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        current_end: int
    ) -> None:
        incoming_len = k.shape[0]

        if current_end <= self.global_end_index:
            write_start_in_cache = self.seq_len - incoming_len

            if write_start_in_cache < 0:
                # This shouldn't happen, but handle gracefully
                self.append(k, v)
                self.global_end_index = current_end
                return
            
            # Calculate page and offset for write_start_in_cache
            start_page_idx = write_start_in_cache // self.page_size
            start_offset = write_start_in_cache % self.page_size

            incoming_processed = 0
            current_page_idx = start_page_idx
            current_offset = start_offset

            while incoming_processed < incoming_len:
                if current_page_idx >= len(self.active_page_indices):
                    # Need more pages than we have - shouldn't happen in overwrite mode
                    break

                page_id = self.active_page_indices[current_page_idx]
                space_in_page = self.page_size - current_offset
                to_write = min(space_in_page, incoming_len - incoming_processed)

                # Overwrite cache
                self.k_cache[
                    page_id,
                    current_offset:current_offset + to_write
                ] = k[incoming_processed:incoming_processed + to_write]

                self.v_cache[
                    page_id,
                    current_offset:current_offset + to_write
                ] = v[incoming_processed:incoming_processed + to_write]

                incoming_processed += to_write
                current_page_idx += 1
                current_offset = 0

            # global_end_index stays the same (we're overwriting, not extending)
        else:
            # Append mode: new position beyond what we've seen
            self.append(k, v)
            self.global_end_index = current_end


    def evict(self, max_allowed_tokens: int) -> int:
        if self.seq_len <= max_allowed_tokens:
            return 0
        
        num_to_remove = self.seq_len - max_allowed_tokens

        pages_to_drop = num_to_remove // self.page_size

        if pages_to_drop <= 0:
            return 0

        sink_pages = math.ceil(self.sink_size / self.page_size)

        max_evictable = len(self.active_page_indices) - sink_pages - 1
        if max_evictable <= 0:
            return 0
        
        if pages_to_drop > 0:
            evicted_page_ids = self.active_page_indices[sink_pages:sink_pages + pages_to_drop]

            self.free_page_pool.extend(evicted_page_ids)

            del self.active_page_indices[sink_pages:sink_pages + pages_to_drop]

            evicted_tokens = pages_to_drop * self.page_size
            self.seq_len -= evicted_tokens
            return evicted_tokens
        
        return 0

    def get_flashinfer_meta(
        self,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate FlashInfer paged attention metadata.

        Args:
            device: Device to place tensors on (defaults to cache device)

        Returns:
            Tuple of (page_indices, indptr, last_page_len):
                - page_indices: [num_pages] int32 tensor of page indices
                - indptr: [2] int32 tensor with [0, num_pages] (batch size = 1)
                - last_page_len: [1] int32 tensor with tokens in last page
        """
        if device is None:
            device = self.device

        # Page indices for active pages
        indices = torch.tensor(
            self.active_page_indices,
            dtype=torch.int32,
            device=device
        )

        # Indptr for single batch: [0, num_active_pages]
        indptr = torch.tensor(
            [0, len(self.active_page_indices)],
            dtype=torch.int32,
            device=device
        )

        # Number of valid tokens in last page
        last_page_len = torch.tensor(
            [self.current_page_offset if self.current_page_offset > 0 else self.page_size],
            dtype=torch.int32,
            device=device
        )

        return indices, indptr, last_page_len
    
    @property
    def total_tokens(self) -> int:
        """Total number of valid tokens in cache."""
        return self.seq_len

    def __repr__(self) -> str:
        return (
            f"PagedCache("
            f"seq_len={self.seq_len}, "
            f"pages={len(self.active_page_indices)}/{self.max_pages}, "
            f"page_size={self.page_size}, "
            f"heads={self.num_heads}, "
            f"head_dim={self.head_dim})"
        )
    
class PagedCacheManager:
    """
    Manager for multiple PagedCache instances (one per transformer layer).

    This class provides a unified interface for creating and managing
    layer-wise paged caches, compatible with the existing CacheManager API.
    """

    def __init__(
        self,
        num_layers: int,
        max_total_tokens: int,
        page_size: int,
        num_heads: int,
        head_dim: int,
        sink_size: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = "cuda" # type: ignore
    ):
        self.num_layers = num_layers
        self.max_total_tokens = max_total_tokens
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sink_size = sink_size
        self.dtype = dtype
        self.device = device

        # Create per-layer caches
        self.caches: list[PagedCache] = [
            PagedCache(
                max_total_tokens=max_total_tokens,
                page_size=page_size,
                num_heads=num_heads,
                head_dim=head_dim,
                sink_size=sink_size,
                dtype=dtype,
                device=device
            )
            for _ in range(num_layers)
        ]

        # FlashInfer workspace (shared across layers)
        self._workspace_buffer: Optional[torch.Tensor] = None
        self._prefill_wrapper: Optional["BatchPrefillWithPagedKVCacheWrapper"] = None

    def reset(self) -> None:
        for cache in self.caches:
            cache.reset()

    def get_cache(self, layer_idx: int) -> PagedCache:
        return self.caches[layer_idx]

    def init_flashinfer_wrapper(self, device: torch.device) -> None:
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                128 * 1024 * 1024,  # 128MB workspace
                dtype=torch.uint8,
                device=device
            )

        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._workspace_buffer,
                kv_layout="NHD"  # [num_pages, page_size, num_heads, head_dim]
            )

    @property
    def prefill_wrapper(self) -> Optional["BatchPrefillWithPagedKVCacheWrapper"]:
        return self._prefill_wrapper

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, idx: int) -> PagedCache:
        return self.caches[idx]
