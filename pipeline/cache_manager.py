from typing import List, Dict, Optional
import torch

from .config import ModelConfig, CacheConfig
from ..models.paged_cache import PagedCache

class CacheManager:

    def __init__(
            self,
            model_config: ModelConfig,
            cache_config: CacheConfig,
            device: torch.device,
            dtype: torch.dtype,
            page_size: int = 16,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.dtype = dtype
        self.page_size = page_size

        self.visual_cache : Optional[List[PagedCache]] = None

    
    def initialize_all_caches(self, batch_size: int = 1) -> None:
        self.visual_cache = self._create_paged_visual_cache(batch_size)

    def _create_paged_visual_cache(self, batch_size: int) -> List[PagedCache]:
        assert batch_size == 1, "PagedCache currently only supports batch_size=1"

        cache_size = self.cache_config.get_visual_cache_size(self.model_config.frame_seq_length)
        num_heads = self.model_config.num_attention_heads
        head_dim = self.model_config.head_dim

        cache = []
        for _ in range(self.model_config.num_transformer_blocks):
            cache.append(PagedCache(
                max_total_tokens=cache_size,
                page_size=self.page_size,
                num_heads=num_heads,
                head_dim=head_dim,
                sink_size=0,  # TODO: Get from config if needed
                dtype=self.dtype,
                device=self.device,
            ))

        return cache
    
    def is_initialized(self) -> bool:
        """Check if caches have been initialized."""
        return self.visual_cache is not None
    
    def reset_all_caches(self) -> None:
        if self.visual_cache is None:
            raise RuntimeError("Caches must be initialized before resetting")

        for block_cache in self.visual_cache:
            block_cache.reset()