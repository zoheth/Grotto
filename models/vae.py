import logging
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_3_t
from torch.nn.modules.utils import _triple
from einops import rearrange

__all__ = ['WanVAE']

CACHE_T = 2  # Frames to cache

class CausalConv3d(nn.Conv3d):
    def __init__(self,
                 in_channels: int,
                out_channels: int,
                kernel_size: _size_3_t,
                stride: _size_3_t = 1,
                padding: Union[str, _size_3_t] = 0,
                dilation: _size_3_t = 1,
                groups: int = 1,
                bias: bool = True,):
        k_t, k_h, k_w = _triple(kernel_size)
        s_t, s_h, s_w = _triple(stride)
        d_t, d_h, d_w = _triple(dilation)
        
        self.causal_padding_t = (k_t - 1) * d_t
        
        padding_h, padding_w = 0, 0
        if isinstance(padding, int):
            padding_h = padding
            padding_w = padding
        else:
            raise NotImplementedError
        
        self.spatial_padding_h = padding_h
        self.spatial_padding_w = padding_w
        
        self.padding_tuple = (
            self.spatial_padding_w, self.spatial_padding_w, # W (dim 4)
            self.spatial_padding_h, self.spatial_padding_h, # H (dim 3)
            self.causal_padding_t, 0                         # T (dim 2) [左填充, 右填充=0]
        )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0, # <<< 核心：禁止父类自动填充
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
    def forward(self, x: torch.Tensor, cache_x: torch.Tensor = None) -> torch.Tensor:
        padding_to_apply = list(self.padding_tuple)
        T_LEFT_INDEX = 4
        
        if cache_x is not None:
            if self.causal_padding_t > 0:
                cache_x = cache_x.to(x.device)
                x = torch.cat([cache_x, x], dim=2)
                padding_needed = max(0, self.causal_padding_t - cache_x.shape[2])
                padding_to_apply[T_LEFT_INDEX] = padding_needed
                
        x_padded = F.pad(x, padding_to_apply)
        
        return super().forward(input)

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

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)