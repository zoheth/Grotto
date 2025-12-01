"""
FlashInfer-based Causal Self-Attention for inference.

Architecture:
- RoPE3DCache: On-demand 3D RoPE frequency computation
- apply_rope_3d: Apply 3D RoPE using precomputed frequencies
- FlashInferPlanner: Manages plan state (once per generation step)
- CausalSelfAttention: Paged attention with FlashInfer
"""

from typing import Optional, Tuple

import flashinfer
import torch
import torch.nn as nn
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from grotto.modeling.paged_cache import PagedCache


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


class FlashInferPlanner:
    """
    Manages FlashInfer plan state for a generation step.

    Plan is executed once per generation step and shared across all layers.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        page_size: int,
        workspace_size: int = 128 * 1024 * 1024,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.workspace_size = workspace_size

        self._workspace_buffer: Optional[torch.Tensor] = None
        self._prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
        self._is_planned = False

    def init(self, device: torch.device) -> None:
        """Initialize FlashInfer workspace and wrapper."""
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                self.workspace_size, dtype=torch.uint8, device=device
            )

        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._workspace_buffer, kv_layout="NHD"
            )

    def plan(
        self,
        kv_cache: PagedCache,
        q_len: int,
        device: torch.device,
        q_dtype: torch.dtype,
    ) -> None:
        """Execute plan for the current generation step."""
        self.init(device)

        paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len = kv_cache.get_flashinfer_meta(
            device
        )
        qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=device)

        assert self._prefill_wrapper is not None

        self._prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            causal=False,
            q_data_type=q_dtype,
        )

        self._is_planned = True

    def run(
        self,
        q: torch.Tensor,  # [q_len, num_heads, head_dim]
        kv_cache: PagedCache,
    ) -> torch.Tensor:
        """Run paged attention using the pre-computed plan."""

        assert self._prefill_wrapper is not None

        return self._prefill_wrapper.run(
            q,
            (kv_cache.k_cache, kv_cache.v_cache),
        )

    @property
    def is_planned(self) -> bool:
        return self._is_planned

    def reset(self) -> None:
        """Reset plan state."""
        self._is_planned = False


class CausalSelfAttention(nn.Module):
    """
    FlashInfer-based causal self-attention with 3D RoPE.

    Inference-only implementation using paged KV cache.

    **Alignment with training block_mask (block-aligned, "staircase" pattern):**

    Training uses blockwise causal mask:
        (kv_idx < ends[q_idx]) & (kv_idx >= ends[q_idx] - local_attn_size * frame_seqlen)
    where ends[q_idx] is the end of the current block.

    Inference implements block-aligned eviction:
    - Determine current block from current_start
    - From current_block_end, look back local_attn_size frames
    - Align down to block boundary (ensures complete blocks only)
    - Example (num_frame_per_block=3, local_attn_size=6):
      * Block 2 (frames [6,7,8]): keeps [3,4,5,6,7,8] (Block 1 + Block 2)
      * When generating frame 8: keeps [3,4,5,6,7] + current tokens
      * When generating frame 9: keeps [6,7,8] + current tokens
    - This creates the "staircase" pattern: only complete previous blocks visible
    """

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
        page_size: int = 256,  # Increased from 16 to reduce d2d copies (880 tokens/frame ÷ 256 = ~4 copies)
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
        self.page_size = page_size

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.rope_cache: Optional[RoPE3DCache] = None

    def _init_rope_cache(self, freqs: torch.Tensor):
        """Initialize RoPE cache."""
        if self.rope_cache is None:
            self.rope_cache = RoPE3DCache(
                freqs=freqs,
                height=self.height,
                width=self.width,
            )

    def forward(
        self,
        x: torch.Tensor,  # [B, seq_len, dim]
        grid_sizes: Tuple[int, int, int],  # (F, H, W)
        freqs: torch.Tensor,
        kv_cache: PagedCache,
        current_start: int,
        planner: FlashInferPlanner,
    ) -> torch.Tensor:
        """
        Forward pass with paged attention.

        First layer plans after cache update, later layers reuse the plan.
        """
        B, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert B == 1, "PagedCache only supports batch size 1"

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

        # FlashInfer RoPE with cos/sin cache
        # Cache format: [max_pos, rotary_dim] where first half is cos, second half is sin
        # Use is_neox=False for interleaved rotation (matches original apply_rope_3d)
        roped_q_flat, roped_k_flat = flashinfer.apply_rope_with_cos_sin_cache(
            positions,
            q_flat,
            k_flat,
            head_size=d,
            cos_sin_cache=cache,
            is_neox=False,  # Interleaved format like original implementation
        )
        # Restore shape with batch dimension [B, s, n, d]
        roped_q = roped_q_flat.view(B, s, self.num_heads, self.head_dim)
        roped_k = roped_k_flat.view(B, s, self.num_heads, self.head_dim)

        # Ensure tensors are contiguous before cache write to avoid extra d2d copies
        roped_k_squeezed = roped_k.squeeze(0).contiguous()
        v_squeezed = v.squeeze(0).contiguous()
        current_end = current_start + roped_k_squeezed.shape[0]

        kv_cache.update_or_append(roped_k_squeezed, v_squeezed, current_end)

        # Block-aligned eviction to match training block_mask
        block_size = self.num_frame_per_block * frame_seqlen
        if self.local_attn_size == -1:
            # Global attention: keep all frames (capped at 15 for memory)
            keep_size = 15 * frame_seqlen
        else:
            # Calculate which block current_start belongs to
            current_block_idx = current_start // block_size
            # Calculate the end of current block (block boundary)
            current_block_end = (current_block_idx + 1) * block_size

            # From current_block_end, look back local_attn_size frames (matches training)
            keep_from_position = max(0, current_block_end - self.local_attn_size * frame_seqlen)

            # Align down to block boundary for "staircase" pattern
            # keep_from_block_idx = keep_from_position // block_size
            # keep_from_position = keep_from_block_idx * block_size

            # Calculate how many tokens to keep
            keep_size = current_end - keep_from_position

        kv_cache.evict(keep_size)

        if not planner.is_planned:
            planner.plan(
                kv_cache=kv_cache,
                q_len=roped_q.shape[1],
                device=roped_q.device,
                q_dtype=roped_q.dtype,
            )

        q_for_flash = roped_q.squeeze(0)
        x = planner.run(q_for_flash, kv_cache)
        x = x.unsqueeze(0)

        x = x.flatten(2)
        x = self.o(x)
        return x
