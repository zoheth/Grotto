from typing import Optional

import flashinfer
import torch
import torch.nn as nn

from grotto.modeling.kv_cache import DualPlaneKVCache


class AttentionWithCache(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_frame_per_block: int,
        max_frames: int,  # frame_seq_len * num_frame_per_block
        local_attn_size: int = -1,
        workspace_buffer: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_frame_per_block = num_frame_per_block
        self.local_attn_size = local_attn_size

        # Buffer initialization
        if workspace_buffer is None:
            workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        self.kv_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
        self.qo_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")

        # Initialize qo_indptr based on max query length logic from original code
        self.qo_indptr[1] = max_frames

        self.flashinfer_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer,
            "NHD",
        )

    def plan(
        self,
        incoming_len: int,
        kv_cache: DualPlaneKVCache,
        current_start: int,
        current_end: int,
        frame_seqlen: int,  # height * width
        cache_mode: str = "read_write",
    ) -> None:
        """
        Plan phase: All CPU operations.
        Updates KV Cache state and plans FlashInfer execution.
        """
        if cache_mode == "read_write":
            # 1. Plan append
            kv_cache.plan_append(incoming_len)

            # 2. Plan eviction
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
            # Read-only mode: calculate temporary length
            kv_len = kv_cache.total_tokens + incoming_len
        else:
            raise ValueError(f"Invalid cache_mode: {cache_mode}")

        # 3. Plan FlashInfer
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
        roped_q: torch.Tensor,
        roped_k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: DualPlaneKVCache,
        cache_mode: str = "read_write",
    ) -> torch.Tensor:
        """
        Execute phase: GPU operations.
        Assumes roped_q and roped_k have (S, H, D) shape.
        """

        # 1. Execute KV cache operations
        if cache_mode == "read_only":
            read_plan = kv_cache.get_read_plan()
            k_linear, v_linear = kv_cache._storage.execute_gather_with_append(read_plan, roped_k, v)
        elif cache_mode == "read_write":
            kv_cache.execute_append(roped_k, v)
            k_linear, v_linear = kv_cache.get_linear_view()
        else:
            # Should be caught in plan(), but double check safety
            raise ValueError(f"Invalid cache_mode: {cache_mode}")

        # 2. Run FlashInfer attention
        q_for_flash = roped_q.squeeze(0)
        x = self.flashinfer_wrapper.run(q_for_flash, k_linear, v_linear)

        # Restore batch dimension
        x = x.unsqueeze(0)
        return x
