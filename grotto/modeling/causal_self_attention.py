"""FlashInfer-based self-attention with RingBuffer KV cache and 3D RoPE."""

from typing import Optional, Tuple

import flashinfer
import torch
import torch.nn as nn

from grotto.modeling.ring_buffer_visual_cache import RingBufferVisualCache


class RoPE3DCache:
    def __init__(
        self,
        freqs: torch.Tensor,
        height: int,
        width: int,
        max_frames: int = 150,
    ):
        device = freqs.device

        # If freqs is complex (e^(iθ)), extract real and imaginary parts
        # Otherwise assume it's already angles and convert to complex first
        if freqs.is_complex():
            freqs = freqs.to(device=device)
        else:
            freqs = freqs.to(dtype=torch.float32, device=device)

        head_dim_half = freqs.shape[1]
        c_height = head_dim_half // 3
        c_width = head_dim_half // 3
        c_time = head_dim_half - c_height - c_width

        freqs_time = freqs[:max_frames, :c_time]
        freqs_height = freqs[:height, c_time : c_time + c_height]
        freqs_width = freqs[:width, c_time + c_height :]

        t_grid = freqs_time.view(max_frames, 1, 1, -1).expand(max_frames, height, width, -1)
        h_grid = freqs_height.view(1, height, 1, -1).expand(max_frames, height, width, -1)
        w_grid = freqs_width.view(1, 1, width, -1).expand(max_frames, height, width, -1)

        flat_freqs = torch.cat([t_grid, h_grid, w_grid], dim=-1).reshape(-1, head_dim_half)

        # If freqs is complex (e^(iθ)), extract cos (real) and sin (imag) directly
        # Otherwise compute cos and sin from angles
        if flat_freqs.is_complex():
            cos = flat_freqs.real.float()
            sin = flat_freqs.imag.float()
        else:
            cos = torch.cos(flat_freqs)
            sin = torch.sin(flat_freqs)

        # FlashInfer expects: [max_pos, rotary_dim] where first half is cos, second half is sin
        self.global_cache = torch.cat([cos, sin], dim=-1).contiguous()

    def get_cache(self) -> torch.Tensor:
        return self.global_cache


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        qk_norm: bool = True,
        eps: float = 1e-6,
        height: int = 22,
        width: int = 40,
    ):
        super().__init__()

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block

        self.height = height
        self.width = width
        self.frame_seq_len = height * width

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.rope_cache: Optional[RoPE3DCache] = None

    def _init_rope_cache(self, freqs: torch.Tensor):
        if self.rope_cache is None:
            self.rope_cache = RoPE3DCache(freqs=freqs, height=self.height, width=self.width)

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: Tuple[int, int, int],
        freqs: torch.Tensor,
        kv_cache: RingBufferVisualCache,
        current_start: int,
    ) -> torch.Tensor:
        B, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert B == 1, "RingBufferVisualCache only supports batch size 1"

        q = self.norm_q(self.q(x)).view(B, s, n, d)
        k = self.norm_k(self.k(x)).view(B, s, n, d)
        v = self.v(x).view(B, s, n, d)

        self._init_rope_cache(freqs)

        assert self.rope_cache is not None

        _, height, width = grid_sizes
        frame_seqlen = height * width

        nnz = B * s
        q_flat = q.view(nnz, n * d)
        k_flat = k.view(nnz, n * d)
        positions = torch.arange(
            current_start, current_start + nnz, dtype=torch.int32, device=x.device
        )

        cache = self.rope_cache.get_cache()

        roped_q_flat, roped_k_flat = flashinfer.apply_rope_with_cos_sin_cache(
            positions, q_flat, k_flat, head_size=d, cos_sin_cache=cache, is_neox=False
        )
        roped_q = roped_q_flat.view(B, s, self.num_heads, self.head_dim)
        roped_k = roped_k_flat.view(B, s, self.num_heads, self.head_dim)

        roped_k_squeezed = roped_k.squeeze(0).contiguous()
        v_squeezed = v.squeeze(0).contiguous()
        current_end = current_start + roped_k_squeezed.shape[0]

        kv_cache.update_or_append(roped_k_squeezed, v_squeezed, current_end)
        block_size = self.num_frame_per_block * frame_seqlen
        if self.local_attn_size == -1:
            keep_size = 15 * frame_seqlen
        else:
            current_block_idx = current_start // block_size
            current_block_end = (current_block_idx + 1) * block_size
            keep_from_position = max(0, current_block_end - self.local_attn_size * frame_seqlen)
            keep_size = current_end - keep_from_position

        kv_cache.evict(keep_size)
        k_cache, v_cache = kv_cache.get_kv_cache()

        q_for_flash = roped_q.squeeze(0)
        x = flashinfer.single_prefill_with_kv_cache(
            q_for_flash, k_cache, v_cache, causal=False, kv_layout="NHD"
        )
        x = x.unsqueeze(0)

        x = x.flatten(2)
        x = self.o(x)
        return x
