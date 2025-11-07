import logging
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['WanVAE']

CACHE_T = 2  # Frames to cache

class CacheManager:
    """
    Centralized cache manager for all causal convolutions.

    Design: Each CausalConv3d layer auto-registers and auto-manages its cache.
    """

    def __init__(self):
        self.caches: List[Optional[torch.Tensor]] = []
        self.index = 0
        self.enabled = False

    def enable(self, num_layers: int):
        self.caches = [None] * num_layers
        self.index = 0
        self.enabled = True

    def disable(self):
        self.caches = []
        self.index = 0
        self.enabled = False

    def reset(self):
        self.index = 0

    def next_cache(self) -> Optional[torch.Tensor]:
        if not self.enabled or self.index >= len(self.caches):
            return None
        cache = self.caches[self.index]
        self.index += 1
        return cache
    
    def update_cache(self, new_cache: torch.Tensor):
        if self.enabled and self.index > 0:
            self.caches[self.index - 1] = new_cache

class CausalConv3d(nn.Module):
    _cache_manager: Optional[CacheManager] = None

    @classmethod
    def set_cache_manager(cls, manager: Optional[CacheManager]):
        """Set global cache manager."""
        cls._cache_manager = manager

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding, self.padding)
            
        self.causal_padding_t = self.padding[0]

        self._padding = (
            self.padding[2], self.padding[2],  # W
            self.padding[1], self.padding[1],  # H
            2 * self.padding[0], 0              # T (causal)
        )

        # 重置父类的 padding，因为我们使用 F.pad 手动管理
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, 
                  cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        显式管理缓存的 forward。

        Args:
            x (torch.Tensor): 输入张量 [B, C, T, H, W]。
                               在自回归（autoregressive）模式下, T 可能是 1。
            cache (Optional[torch.Tensor]): 
                               来自上一步的缓存 [B, C, P, H, W]，
                               其中 P == self.causal_padding_t * 2。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - output: 卷积后的输出张量
            - new_cache: 用于下一步的新缓存
        """

        P = self._padding[4] # 时间维度的前填充大小

        if x.shape[2] < P:
            if cache is not None:
                shortfall = P - x.shape[2]
                prev_frames = cache[:, :, -shortfall:, :, :]
                new_x = torch.cat([prev_frames, x], dim=2)



class Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temporal_upsample: List[bool] = [False, True, True],
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_upsample = temporal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = 