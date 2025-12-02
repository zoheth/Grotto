from typing import List

import torch

from grotto.modeling.kv_cache import DualPlaneKVCache
from grotto.modeling.ring_buffer_cache import RingBufferActionCache
from grotto.pipeline.config import CacheConfig, ModelConfig


class CacheManager:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.dtype = dtype

        self.visual_cache: List[DualPlaneKVCache]
        self.mouse_cache: List[RingBufferActionCache]
        self.keyboard_cache: List[RingBufferActionCache]

    def initialize_all_caches(self, batch_size: int = 1) -> None:
        assert batch_size == 1, "DualPlaneKVCache currently only supports batch_size=1"
        self.visual_cache = self._create_dual_plane_kv_cache(batch_size)

        self.mouse_cache = self._create_action_mouse_cache(batch_size)
        self.keyboard_cache = self._create_action_keyboard_cache(batch_size)

    def _create_dual_plane_kv_cache(self, batch_size: int) -> List[DualPlaneKVCache]:
        assert batch_size == 1, "DualPlaneKVCache currently only supports batch_size=1"

        cache_size = self.cache_config.get_visual_cache_size(self.model_config.frame_seq_length)
        num_heads = self.model_config.num_attention_heads
        head_dim = self.model_config.head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append(
                DualPlaneKVCache(
                    max_seq_len=cache_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        return cache

    def _create_action_mouse_cache(self, batch_size: int) -> List[RingBufferActionCache]:
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
    ) -> tuple[List[DualPlaneKVCache], List[RingBufferActionCache], List[RingBufferActionCache]]:
        if self.visual_cache is None or self.mouse_cache is None or self.keyboard_cache is None:
            raise RuntimeError("Caches must be initialized before access")
        return (self.visual_cache, self.mouse_cache, self.keyboard_cache)
