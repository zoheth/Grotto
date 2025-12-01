from typing import List

import torch

from grotto.modeling.paged_cache import PagedCache
from grotto.modeling.ring_buffer_cache import RingBufferActionCache
from grotto.pipeline.config import CacheConfig, ModelConfig


class CacheManager:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        device: torch.device,
        dtype: torch.dtype,
        page_size: int = 256,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.dtype = dtype
        self.page_size = page_size

        self.visual_cache: List[PagedCache]
        self.mouse_cache: List[RingBufferActionCache]
        self.keyboard_cache: List[RingBufferActionCache]

    def initialize_all_caches(self, batch_size: int = 1) -> None:
        assert batch_size == 1, "PagedCache currently only supports batch_size=1"
        self.visual_cache = self._create_paged_visual_cache(batch_size)

        self.mouse_cache = self._create_action_mouse_cache(batch_size)
        self.keyboard_cache = self._create_action_keyboard_cache(batch_size)

    def _create_paged_visual_cache(self, batch_size: int) -> List[PagedCache]:
        assert batch_size == 1, "PagedCache currently only supports batch_size=1"

        cache_size = self.cache_config.get_visual_cache_size(self.model_config.frame_seq_length)
        num_heads = self.model_config.num_attention_heads
        head_dim = self.model_config.head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append(
                PagedCache(
                    max_total_tokens=cache_size,
                    page_size=self.page_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    sink_size=0,  # TODO: Get from config if needed
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        return cache

    def _create_action_mouse_cache(self, batch_size: int) -> List[RingBufferActionCache]:
        """
        Create RingBufferActionCache for mouse action conditioning.

        Mouse self-attention uses spatial batching [B*S, T, H, D] where S is the
        number of spatial tokens (frame_seq_length). RingBufferActionCache pre-allocates
        memory for the full batch dimension.

        Uses Ring Buffer for CUDA Graph compatibility.

        Args:
            batch_size: Base batch size (will be multiplied by spatial dimension)

        Returns:
            List of RingBufferActionCache instances, one per transformer block
        """
        # Mouse cache needs B * S where S is spatial dimension
        mouse_batch_size = batch_size * self.model_config.frame_seq_length

        cache_size = self.cache_config.get_action_cache_size()
        num_heads = self.model_config.num_action_attention_heads
        head_dim = self.model_config.action_head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append(
                RingBufferActionCache(
                    batch_size=mouse_batch_size,
                    max_seq_len=cache_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        return cache

    def _create_action_keyboard_cache(self, batch_size: int) -> List[RingBufferActionCache]:
        """
        Create RingBufferActionCache for keyboard action conditioning.

        Keyboard self-attention uses simple batching [B, T, H, D].
        RingBufferActionCache pre-allocates memory for the batch dimension.

        Uses Ring Buffer for CUDA Graph compatibility.

        Args:
            batch_size: Batch size

        Returns:
            List of RingBufferActionCache instances, one per transformer block
        """
        cache_size = self.cache_config.get_action_cache_size()
        num_heads = self.model_config.num_action_attention_heads
        head_dim = self.model_config.action_head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append(
                RingBufferActionCache(
                    batch_size=batch_size,
                    max_seq_len=cache_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        return cache

    def is_initialized(self) -> bool:
        """Check if caches have been initialized."""
        return len(self.visual_cache) > 0

    def reset_all_caches(self) -> None:
        for block_cache in self.visual_cache:
            block_cache.reset()

        for block_cache in self.mouse_cache:
            block_cache.reset()

        for block_cache in self.keyboard_cache:
            block_cache.reset()

    def get_caches(
        self,
    ) -> tuple[List[PagedCache], List[RingBufferActionCache], List[RingBufferActionCache]]:
        """
        Get all caches.

        Returns:
            Tuple of (visual_cache, mouse_cache, keyboard_cache, None)
            - visual_cache: PagedCache instances (page-granular eviction)
            - mouse_cache: RingBufferActionCache instances (ring buffer, spatial batching)
            - keyboard_cache: RingBufferActionCache instances (ring buffer, simple batching)
        """
        if self.visual_cache is None or self.mouse_cache is None or self.keyboard_cache is None:
            raise RuntimeError("Caches must be initialized before access")

        return (self.visual_cache, self.mouse_cache, self.keyboard_cache)
