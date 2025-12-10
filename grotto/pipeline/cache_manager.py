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

    def initialize_all_caches(self, batch_size: int = 1) -> None:
        assert batch_size == 1
        self.visual_cache = self._create_dual_plane_kv_cache()

    def _create_dual_plane_kv_cache(self) -> List[DualPlaneKVCache]:
        cache_size = self.cache_config.get_visual_cache_size(self.model_config.frame_seq_length)
        num_heads = self.model_config.num_attention_heads
        head_dim = self.model_config.head_dim
        incoming_len = self.model_config.frame_seq_length * 3

        return [
            DualPlaneKVCache(
                max_seq_len=cache_size,
                max_incoming_len=incoming_len,
                num_heads=num_heads,
                head_dim=head_dim,
                tokens_per_latent=incoming_len,
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

    def get_caches(self) -> List[DualPlaneKVCache]:
        if self.visual_cache is None:
            raise RuntimeError("Caches must be initialized before access")
        return self.visual_cache
