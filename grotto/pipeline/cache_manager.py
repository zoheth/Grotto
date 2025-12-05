from typing import List

import torch

from grotto.modeling.kv_cache import DualPlaneKVCache
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
        self.mouse_cache: List[DualPlaneKVCache]
        self.keyboard_cache: List[DualPlaneKVCache]

    def initialize_all_caches(self, batch_size: int = 1) -> None:
        assert batch_size == 1
        self.visual_cache = self._create_dual_plane_kv_cache()
        self.mouse_cache = self._create_action_mouse_cache()
        self.keyboard_cache = self._create_action_keyboard_cache()

    def _create_dual_plane_kv_cache(self) -> List[DualPlaneKVCache]:
        cache_size = self.cache_config.get_visual_cache_size(self.model_config.frame_seq_length)
        num_heads = self.model_config.num_attention_heads
        head_dim = self.model_config.head_dim

        return [
            DualPlaneKVCache(
                max_seq_len=cache_size,
                max_incoming_len=self.model_config.frame_seq_length,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.model_config.num_transformer_blocks)
        ]

    def _create_action_mouse_cache(self) -> List[DualPlaneKVCache]:
        # mouse is self-attention
        cache_size = self.cache_config.get_visual_cache_size(self.model_config.frame_seq_length)
        num_heads = self.model_config.num_action_attention_heads
        head_dim = self.model_config.action_head_dim

        return [
            DualPlaneKVCache(
                max_seq_len=cache_size,
                max_incoming_len=self.model_config.frame_seq_length,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.model_config.num_transformer_blocks)
        ]

    def _create_action_keyboard_cache(self) -> List[DualPlaneKVCache]:
        cache_size = self.cache_config.get_action_cache_size()
        num_heads = self.model_config.num_action_attention_heads
        head_dim = self.model_config.action_head_dim

        return [
            DualPlaneKVCache(
                max_seq_len=cache_size,
                max_incoming_len=3,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.model_config.num_transformer_blocks)
        ]

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
    ) -> tuple[List[DualPlaneKVCache], List[DualPlaneKVCache], List[DualPlaneKVCache]]:
        if self.visual_cache is None or self.mouse_cache is None or self.keyboard_cache is None:
            raise RuntimeError("Caches must be initialized before access")
        return (self.visual_cache, self.mouse_cache, self.keyboard_cache)
