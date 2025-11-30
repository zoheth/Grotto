"""
FlashInfer-based Causal Self-Attention for inference.

Architecture:
- RoPE3DCache: On-demand 3D RoPE frequency computation
- apply_rope_3d: Apply 3D RoPE using precomputed frequencies
- FlashInferPlanner: Manages plan state (once per generation step)
- CausalSelfAttention: Paged attention with FlashInfer
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from grotto.modeling.paged_cache import PagedCache


class RoPE3DCache:
    """On-demand 3D RoPE frequency computation for efficient inference."""

    def __init__(
        self,
        freqs: torch.Tensor,  # [max_positions, head_dim//2]
        height: int,
        width: int,
    ):
        self.height = height
        self.width = width
        self.device = freqs.device
        self.dtype = freqs.dtype

        head_dim_half = freqs.shape[1]
        c = head_dim_half
        self.c_time = c - 2 * (c // 3)
        self.c_height = c // 3
        self.c_width = c // 3
        self.head_dim_half = head_dim_half

        self.freqs_time = freqs[:, :self.c_time].to(self.device)
        self.freqs_height = freqs[:, self.c_time:self.c_time + self.c_height].to(self.device)
        self.freqs_width = freqs[:, self.c_time + self.c_height:].to(self.device)

    def get_freqs_for_frame_range(
        self,
        start_frame: int,
        num_frames: int
    ) -> torch.Tensor:
        """Compute frequencies for a range of frames on-demand."""
        end_frame = start_frame + num_frames

        if end_frame > self.freqs_time.shape[0]:
            raise ValueError(
                f"Frame range [{start_frame}:{end_frame}] exceeds frequency table size {self.freqs_time.shape[0]}"
            )

        f, h, w = num_frames, self.height, self.width

        time_freqs = self.freqs_time[start_frame:end_frame].view(f, 1, 1, -1).expand(f, h, w, -1)
        height_freqs = self.freqs_height[:h].view(1, h, 1, -1).expand(f, h, w, -1)
        width_freqs = self.freqs_width[:w].view(1, 1, w, -1).expand(f, h, w, -1)

        freqs = torch.cat([time_freqs, height_freqs, width_freqs], dim=-1)
        return freqs.reshape(-1, 1, self.head_dim_half)


def apply_rope_3d(
    x: torch.Tensor,  # [B, seq_len, num_heads, head_dim]
    freqs: torch.Tensor,  # [seq_len, 1, head_dim//2]
) -> torch.Tensor:
    """Apply 3D RoPE using precomputed frequencies."""
    B, seq_len, num_heads, head_dim = x.shape

    x_complex = torch.view_as_complex(
        x.reshape(B, seq_len, num_heads, head_dim // 2, 2).to(torch.float64)
    )

    x_rotated = x_complex * freqs.unsqueeze(0)
    x_out = torch.view_as_real(x_rotated).flatten(-2)

    return x_out.type_as(x)


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
                self.workspace_size,
                dtype=torch.uint8,
                device=device
            )

        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._workspace_buffer,
                kv_layout="NHD"
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

        paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len = kv_cache.get_flashinfer_meta(device)
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
        page_size: int = 16,
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

        num_frames, height, width = grid_sizes
        frame_seqlen = height * width
        current_start_frame = current_start // frame_seqlen

        precomputed_freqs = self.rope_cache.get_freqs_for_frame_range(
            current_start_frame, num_frames
        )

        roped_q = apply_rope_3d(q, precomputed_freqs)
        roped_k = apply_rope_3d(k, precomputed_freqs)

        roped_k_squeezed = roped_k.squeeze(0)
        v_squeezed = v.squeeze(0)
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
