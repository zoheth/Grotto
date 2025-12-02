"""FlashInfer-based self-attention with RingBuffer KV cache and 3D RoPE."""

from typing import Optional, Tuple

import flashinfer
import torch
import torch.nn as nn

from grotto.modeling.kv_cache import DualPlaneKVCache


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
        num_frame_per_block: int,
        local_attn_size: int = -1,
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

        if workspace_buffer is None:
            workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        self.kv_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
        self.qo_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
        q_len = self.frame_seq_len * num_frame_per_block
        self.qo_indptr[1] = q_len

        self.flashinfer_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer,
            "NHD",
        )

        self.register_buffer(
            "local_indices", torch.arange(q_len, dtype=torch.int32), persistent=False
        )

    def _init_rope_cache(self, freqs: torch.Tensor):
        if self.rope_cache is None:
            self.rope_cache = RoPE3DCache(freqs=freqs, height=self.height, width=self.width)

    def plan_kv_and_attention(
        self,
        incoming_len: int,
        kv_cache: DualPlaneKVCache,
        current_start: int,
        current_end: int,
        grid_sizes: Tuple[int, int, int],
        cache_mode: str = "read_write",
    ) -> None:
        """
        Plan phase: All CPU operations, no GPU sync.
        Must be called before forward().

        Args:
            incoming_len: Number of new tokens to append
            kv_cache: DualPlaneKVCache instance
            current_start: Start position of current sequence
            current_end: End position after appending new tokens
            grid_sizes: (num_frames, height, width) tuple
            cache_mode: "read_write" or "read_only"
        """
        _, height, width = grid_sizes
        frame_seqlen = height * width

        if cache_mode == "read_write":
            # 1. Plan append (CPU operation)
            kv_cache.plan_append(incoming_len)

            # 2. Plan eviction (CPU operation - just updates valid_len)
            block_size = self.num_frame_per_block * frame_seqlen
            if self.local_attn_size == -1:
                keep_size = 15 * frame_seqlen
            else:
                current_block_idx = current_start // block_size
                current_block_end = (current_block_idx + 1) * block_size
                keep_from_position = max(0, current_block_end - self.local_attn_size * frame_seqlen)
                keep_size = current_end - keep_from_position

            kv_cache.evict(keep_size)
            kv_len = kv_cache.total_tokens
        elif cache_mode == "read_only":
            # In read_only mode, we don't modify cache, but need correct length for FlashInfer
            # Length = history + new data (temporary)
            kv_len = kv_cache.total_tokens + incoming_len
        else:
            raise ValueError(f"Invalid cache_mode: {cache_mode}")

        # 3. Plan FlashInfer (CPU operation)
        self.kv_indptr[1] = kv_len
        self.flashinfer_wrapper.plan(
            self.qo_indptr,
            self.kv_indptr,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            q_data_type=torch.bfloat16,
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
        """
        Execute phase: Pure GPU operations (planning done externally in batch).

        Args:
            x: Input tensor (B, seq_len, dim)
            grid_sizes: (num_frames, height, width) tuple
            freqs: RoPE frequency tensor
            kv_cache: DualPlaneKVCache instance (must be pre-planned)
            current_start: Start position of current sequence
            cache_mode: "read_write" or "read_only"
            incoming_len: Pre-computed sequence length (unused, kept for compatibility)

        Returns:
            Attention output (B, seq_len, dim)
        """
        B, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert B == 1, "DualPlaneKVCache only supports batch size 1"

        # Note: plan_kv_and_attention() is now called externally in batch before forward pass

        # 1. Compute Q, K, V (GPU)
        q = self.norm_q(self.q(x)).view(B, s, n, d)
        k = self.norm_k(self.k(x)).view(B, s, n, d)
        v = self.v(x).view(B, s, n, d)

        # 2. Apply RoPE (GPU)
        self._init_rope_cache(freqs)
        assert self.rope_cache is not None

        nnz = B * s
        positions = self.local_indices[:nnz] + current_start  # type: ignore

        q_flat = q.view(nnz, n * d)
        k_flat = k.view(nnz, n * d)

        cache = self.rope_cache.get_cache()

        roped_q_flat, roped_k_flat = flashinfer.apply_rope_with_cos_sin_cache(
            positions, q_flat, k_flat, head_size=d, cos_sin_cache=cache, is_neox=False
        )
        roped_q = roped_q_flat.view(B, s, self.num_heads, self.head_dim)
        roped_k = roped_k_flat.view(B, s, self.num_heads, self.head_dim)

        roped_k_squeezed = roped_k.squeeze(0).contiguous()
        v_squeezed = v.squeeze(0).contiguous()

        # 3. Execute KV cache operations (GPU)
        if cache_mode == "read_only":
            # Read-only mode: get history + append new data temporarily (no write to ring)
            read_plan = kv_cache.get_read_plan()
            k_linear, v_linear = kv_cache._storage.execute_gather_with_append(
                read_plan, roped_k_squeezed, v_squeezed
            )
        elif cache_mode == "read_write":
            # Read-write mode: execute append, then get linear view
            kv_cache.execute_append(roped_k_squeezed, v_squeezed)
            k_linear, v_linear = kv_cache.get_linear_view()
        else:
            raise ValueError(
                f"Invalid cache_mode: {cache_mode}. Must be 'read_only' or 'read_write'"
            )

        # 4. Run FlashInfer attention (GPU)
        q_for_flash = roped_q.squeeze(0)
        x = self.flashinfer_wrapper.run(q_for_flash, k_linear, v_linear)
        x = x.unsqueeze(0)

        # 5. Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x
