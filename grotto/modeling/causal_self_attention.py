"""FlashInfer-based self-attention with RingBuffer KV cache and 3D RoPE."""

from typing import Optional, Tuple

import flashinfer
import torch
import torch.nn as nn

from grotto.modeling.attention import AttentionWithCache
from grotto.modeling.kv_cache import DualPlaneKVCache
from grotto.modeling.rope import RoPE3DCache


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_frame_per_block: int,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
        height: int = 22,
        width: int = 40,
        workspace_buffer: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.height = height
        self.width = width
        self.frame_seq_len = height * width

        # Linear Projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # RoPE State
        self.rope_cache: Optional[RoPE3DCache] = None

        # Register buffer for RoPE indices
        q_len = self.frame_seq_len * num_frame_per_block
        self.register_buffer(
            "local_indices", torch.arange(q_len, dtype=torch.int32), persistent=False
        )

        # Attention Backend (FlashInfer + KV Cache Mgmt)
        self.attn_backend = AttentionWithCache(
            num_heads=num_heads,
            head_dim=self.head_dim,
            num_frame_per_block=num_frame_per_block,
            block_seq_len=q_len,
            workspace_buffer=workspace_buffer,
        )

    def _init_rope_cache(self, freqs: torch.Tensor):
        if self.rope_cache is None:
            self.rope_cache = RoPE3DCache(freqs=freqs, height=self.height, width=self.width)

    def plan_kv_and_attention(
        self,
        incoming_len: int,
        kv_cache: DualPlaneKVCache,
        cache_mode: str = "read_write",
    ) -> None:
        """
        Delegates planning to the backend.
        KV cache handles sliding window eviction automatically.
        """
        self.attn_backend.plan(
            incoming_len=incoming_len,
            kv_cache=kv_cache,
            cache_mode=cache_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: Tuple[int, int, int],
        freqs: torch.Tensor,
        kv_cache: DualPlaneKVCache,
        current_start: int,
        cache_mode: str = "read_write",
        incoming_len: Optional[int] = None,
    ) -> torch.Tensor:
        B, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert B == 1, "DualPlaneKVCache only supports batch size 1"

        # 1. Compute Q, K, V
        q = self.norm_q(self.q(x)).view(B, s, n, d)
        k = self.norm_k(self.k(x)).view(B, s, n, d)
        v = self.v(x).view(B, s, n, d)

        # 2. Apply RoPE
        self._init_rope_cache(freqs)
        assert self.rope_cache is not None

        nnz = B * s
        positions = self.local_indices[:nnz] + current_start

        q_flat = q.view(nnz, n * d)
        k_flat = k.view(nnz, n * d)

        cache = self.rope_cache.get_cache()

        roped_q_flat, roped_k_flat = flashinfer.apply_rope_with_cos_sin_cache(
            positions, q_flat, k_flat, head_size=d, cos_sin_cache=cache, is_neox=False
        )
        roped_q = roped_q_flat.view(B * s, self.num_heads, self.head_dim)
        roped_k = roped_k_flat.view(B * s, self.num_heads, self.head_dim)
        v = v.squeeze(0).contiguous()

        # 3. Delegate Attention & Cache Ops to Backend
        x = self.attn_backend(
            roped_q=roped_q, roped_k=roped_k, v=v, kv_cache=kv_cache, cache_mode=cache_mode
        )

        # 4. Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x
