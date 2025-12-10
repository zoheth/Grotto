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
        block_seq_len: int,  # frame_seq_len * num_frame_per_block
        workspace_buffer: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_frame_per_block = num_frame_per_block

        # Buffer initialization
        if workspace_buffer is None:
            workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        self.kv_indptr = torch.zeros(2, dtype=torch.int32, device="cpu")
        self.qo_indptr = torch.zeros(2, dtype=torch.int32, device="cpu")

        # Initialize qo_indptr based on max query length logic from original code
        self.qo_indptr[1] = block_seq_len

        self.flashinfer_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer,
            "NHD",
        )

    def plan(
        self,
        incoming_len: int,
        kv_cache: DualPlaneKVCache,
        cache_mode: str = "read_write",
    ) -> None:
        """
        Plan phase: All CPU operations.
        Updates KV Cache state and plans FlashInfer execution.
        KV cache automatically handles sliding window eviction via ring buffer.
        """
        if cache_mode == "read_write":
            # 1. Plan append (KV cache automatically handles sliding window via ring buffer)
            kv_cache.plan_append(incoming_len)
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
        print(k_linear.shape)
        # 2. Run FlashInfer attention
        q_for_flash = roped_q.squeeze(0)
        x = self.flashinfer_wrapper.run(q_for_flash, k_linear, v_linear)

        # Restore batch dimension
        x = x.unsqueeze(0)
        return x
