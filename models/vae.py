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


CACHE_T = 2

class CacheState:
    """Manages temporal caching for streaming video decoding."""

    def __init__(self, size):
        self.feat_map: List[Optional[torch.Tensor]] = [None] * size
        self.idx = 0

    def reset_index(self):
        self.idx = 0

    def get_and_increment(self):
        idx = self.idx
        self.idx += 1
        return idx

    def increment_index(self):
        self.idx += 1

class CausalConv3d(nn.Conv3d):
    """3D convolution with causal temporal padding for autoregressive video generation."""

    def __init__(self,
                 in_channels: int,
                out_channels: int,
                kernel_size: _size_3_t,
                stride: _size_3_t = 1,
                padding: Union[str, _size_3_t] = 0,
                dilation: _size_3_t = 1,
                groups: int = 1,
                bias: bool = True,):
        k_t, _, _ = _triple(kernel_size)
        d_t, _, _ = _triple(dilation)

        self.causal_padding_t = (k_t - 1) * d_t

        padding_h, padding_w = 0, 0
        if isinstance(padding, int):
            padding_h = padding
            padding_w = padding
        elif isinstance(padding, (tuple, list)):
            padding_tuple = _triple(padding)
            _, padding_h, padding_w = padding_tuple
        else:
            raise NotImplementedError(f"Unsupported padding type: {type(padding)}")

        self.spatial_padding_h = padding_h
        self.spatial_padding_w = padding_w

        self.padding_tuple = (
            self.spatial_padding_w, self.spatial_padding_w,
            self.spatial_padding_h, self.spatial_padding_h,
            self.causal_padding_t, 0  # Only left padding for temporal causality
        )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x: torch.Tensor, cache_input: Optional[Union[CacheState, torch.Tensor]] = None) -> torch.Tensor: # type: ignore
        if cache_input is None or self.causal_padding_t == 0:
            return super().forward(F.pad(x, self.padding_tuple))

        if isinstance(cache_input, torch.Tensor):
            cache_tensor = cache_input
            padding_to_apply = list(self.padding_tuple)
            T_LEFT_INDEX = 4

            if cache_tensor is not None and self.causal_padding_t > 0:
                cache_tensor = cache_tensor.to(x.device)
                x = torch.cat([cache_tensor, x], dim=2)
                padding_to_apply[T_LEFT_INDEX] = max(0, self.causal_padding_t - cache_tensor.shape[2])

            x_padded = F.pad(x, padding_to_apply)
            return super().forward(x_padded)

        cache_state = cache_input
        idx = cache_state.get_and_increment()
        past_cache = cache_state.feat_map[idx]

        new_cache = x[:, :, -self.causal_padding_t:, :, :].clone()
        if new_cache.shape[2] < self.causal_padding_t and past_cache is not None:
            needed = self.causal_padding_t - new_cache.shape[2]
            new_cache = torch.cat([
                past_cache[:, :, -needed:, :, :].to(new_cache.device),
                new_cache
            ], dim=2)

        cache_state.feat_map[idx] = new_cache

        padding_to_apply = list(self.padding_tuple)
        T_LEFT_INDEX = 4

        if past_cache is not None:
            if self.causal_padding_t > 0:
                past_cache = past_cache.to(x.device)
                x = torch.cat([past_cache, x], dim=2)
                padding_needed = max(0, self.causal_padding_t - past_cache.shape[2])
                padding_to_apply[T_LEFT_INDEX] = padding_needed

        x_padded = F.pad(x, padding_to_apply)
        return super().forward(x_padded)

class RmsNorm(nn.Module):
    """Root Mean Square normalization with learnable scale and optional bias."""

    def __init__(self, dim: int, channel_first: bool = True, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else 0.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_dim = 1 if self.channel_first else -1
        assert x.shape[norm_dim] == self.dim, \
            f"Input dim {x.shape[norm_dim]} != expected {self.dim}"

        x_norm = F.normalize(x, dim=norm_dim) * self.scale

        gamma_to_apply = self.gamma
        bias_to_apply = self.bias

        if self.channel_first:
            param_shape = [1] * x.ndim
            param_shape[norm_dim] = self.dim
            gamma_to_apply = self.gamma.view(*param_shape)
            if isinstance(self.bias, torch.Tensor):
                bias_to_apply = self.bias.view(*param_shape)

        return x_norm * gamma_to_apply + bias_to_apply

    def __repr__(self):
        return (f"RmsNorm(dim={self.dim}, channel_first={self.channel_first}, "
                f"bias={isinstance(self.bias, nn.Parameter)})")


class ResidualBlock(nn.Module):
    """Residual block with two causal convolutions and skip connection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.norm1 = RmsNorm(in_dim)
        self.silu1 = nn.SiLU()
        self.conv1 = CausalConv3d(in_dim, out_dim, kernel_size=3, padding=1)

        self.norm2 = RmsNorm(out_dim)
        self.silu2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_dim, out_dim, kernel_size=3, padding=1)

        self.shortcut = CausalConv3d(in_dim, out_dim, kernel_size=1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, cache_state: Optional[CacheState] = None) -> torch.Tensor:
        h = self.shortcut(x)

        x = self.norm1(x)
        x = self.silu1(x)
        x = self.conv1(x, cache_state)

        x = self.norm2(x)
        x = self.silu2(x)
        x = self.dropout(x)
        x = self.conv2(x, cache_state)

        return h + x

class SpatialAttentionBlock(nn.Module):
    """Spatial self-attention applied independently per frame."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.norm = RmsNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1)

        nn.init.zeros_(self.proj_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)

        qkv = rearrange(self.to_qkv(x), 'bt c3 h w -> bt 1 (h w) c3')
        q, k, v = qkv.chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, 'bt 1 (h w) c -> bt c h w', h=h, w=w)

        x = self.proj_out(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', b=b, t=t)
        return res + x

class Upsample(nn.Upsample):
    """Upsample with bfloat16 support fix."""

    def forward(self, x): # type: ignore
        return super().forward(x.float()).type_as(x)

class Resample(nn.Module):
    """Spatial or spatiotemporal upsampling with optional temporal interpolation."""

    def __init__(self, dim: int, mode: str = 'upsample3d'):
        assert mode in ['upsample2d', 'upsample3d'], \
            f"Unsupported resample mode: {mode}"
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(dim, dim//2, kernel_size=3, padding=1)
            )
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(dim, dim//2, kernel_size=3, padding=1)
            )
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

            # Stream state: 0=first frame, 1=second frame, 2=running
            self.register_buffer(
                'stream_state', torch.tensor(0, dtype=torch.int)
            )

    def forward(self, x: torch.Tensor, cache_state: Optional[CacheState] = None) -> torch.Tensor:
        b, c, t, h, w = x.shape

        if self.mode == 'upsample2d':
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.resample(x)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
            return x
        elif self.mode == 'upsample3d':
            if cache_state is not None:
                idx = cache_state.idx
                past_cache = cache_state.feat_map[idx]

                new_cache = x[:, :, -CACHE_T:, :, :].clone()
                if new_cache.shape[2] < CACHE_T:
                    if self.stream_state == 0:
                        new_cache = torch.cat([
                            torch.zeros_like(new_cache).to(new_cache.device), new_cache
                        ], dim=2)
                    elif self.stream_state == 1:
                        new_cache = torch.cat([
                            torch.zeros_like(new_cache).to(new_cache.device),
                            new_cache
                        ], dim=2)
                    else:  # stream_state >= 2
                        if past_cache is not None:
                            new_cache = torch.cat([
                                past_cache[:, :, -1, :, :].unsqueeze(2).to(new_cache.device),
                                new_cache
                            ], dim=2)
                        else:
                            new_cache = torch.cat([
                                torch.zeros_like(new_cache).to(new_cache.device),
                                new_cache
                            ], dim=2)

                cache_state.feat_map[idx] = new_cache

                if self.stream_state == 0:
                    cache_state.increment_index()
                    self.stream_state.fill_(1)  # type: ignore
                else:
                    if self.stream_state == 1:
                        x = self.time_conv(x, None)
                        self.stream_state.fill_(2)  # type: ignore
                    else:
                        x = self.time_conv(x, past_cache)

                    cache_state.increment_index()
                    x = rearrange(x, 'b (n c) t h w -> b c (t n) h w', n=2)

            t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x

class VaeDecoder3d(nn.Module):
    """3D VAE decoder with causal convolutions for streaming video generation."""

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

        self.middle = nn.ModuleList([
            ResidualBlock(dims[0], dims[0], dropout),
            SpatialAttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout)
        ])

        self.upsamples = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                self.upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    self.upsamples.append(SpatialAttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temporal_upsample[i] else 'upsample2d'
                self.upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0

        self.head_norm = RmsNorm(dims[-1])
        self.head_silu = nn.SiLU()
        self.head_conv = CausalConv3d(dims[-1], 3, 3, padding=1)

    def forward(self, x: torch.Tensor, cache_states: Optional[List[CacheState]] = None) -> torch.Tensor:
        """
        Forward pass with optional caching for streaming inference.

        Args:
            x: Latent tensor of shape (B, z_dim, T, H, W)
            cache_states: List of 4 CacheState objects for [conv1, middle, upsamples, head_conv]

        Returns:
            Decoded video tensor of shape (B, 3, T', H', W')
        """
        cache_conv1 = cache_states[0] if cache_states is not None else None
        cache_middle = cache_states[1] if cache_states is not None else None
        cache_upsamples = cache_states[2] if cache_states is not None else None
        cache_head = cache_states[3] if cache_states is not None and len(cache_states) > 3 else None

        x = self.conv1(x, cache_conv1)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                x = layer(x, cache_middle)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if isinstance(layer, ResidualBlock):
                x = layer(x, cache_upsamples)
            elif isinstance(layer, Resample):
                x = layer(x, cache_upsamples)
            else:
                x = layer(x)

        x = self.head_norm(x)
        x = self.head_silu(x)
        x = self.head_conv(x, cache_head)

        return x
