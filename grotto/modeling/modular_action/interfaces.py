from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import nn

if TYPE_CHECKING:
    import flashinfer

import flashinfer


class ActionInjector(nn.Module, ABC):
    """Base class for action condition injectors."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        is_causal: bool = False,
        kv_cache=None,
        start_frame: int = 0,
        num_frame_per_block: int = 1,
    ) -> torch.Tensor:
        pass


class AttentionKernel(nn.Module):
    """
    FlashInfer-based attention implementation with integrated RoPE support.

    This implementation can either:
    1. Apply RoPE internally using flashinfer.rope.apply_rope (more efficient)
    2. Accept pre-rotated Q/K tensors (backward compatible)
    """

    def __init__(
        self, rope_scale: float = 1.0, rope_theta: float = 10000.0, interleave: bool = False
    ):
        super().__init__()
        # Wrapper for batch prefill (created lazily on first use)
        self._batch_wrapper = None
        self._workspace_buffer = None

        # RoPE parameters
        self.rope_scale = rope_scale
        self.rope_theta = rope_theta
        self.interleave = interleave

    def _apply_rope_internal(
        self, q: torch.Tensor, k: torch.Tensor, rope_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE using FlashInfer's efficient kernel.

        This method can handle Q and K with the same batch size.
        For cross-attention where Q and K have different batch sizes,
        use _apply_rope_single for each tensor separately.

        Args:
            q: [BS, seq_len_q, num_heads, head_dim] - Query tensor
            k: [BS, seq_len_k, num_heads, head_dim] - Key tensor (must have same BS as q)
            rope_offset: Position offset for RoPE

        Returns:
            Tuple of rotated (q, k) tensors with same shapes
        """
        BS_q, seq_len_q, num_heads, head_dim = q.shape
        BS_k, seq_len_k, _, _ = k.shape

        assert (
            BS_q == BS_k
        ), f"Q and K must have same batch size for joint RoPE application. Got {BS_q} vs {BS_k}"

        # Flatten to ragged format: [BS, L, H, D] -> [BS*L, H, D]
        q_ragged = q.reshape(BS_q * seq_len_q, num_heads, head_dim)
        k_ragged = k.reshape(BS_k * seq_len_k, num_heads, head_dim)

        # Create ragged tensor indices for RoPE
        indptr = torch.arange(
            0, (BS_q + 1) * seq_len_q, seq_len_q, dtype=torch.int32, device=q.device
        )
        # Position offsets for each sequence in the batch
        offsets = torch.full((BS_q,), rope_offset, dtype=torch.int32, device=q.device)

        # Apply RoPE using flashinfer's efficient kernel
        q_ragged, k_ragged = flashinfer.rope.apply_rope(  # type: ignore
            q_ragged,
            k_ragged,
            indptr=indptr,
            offsets=offsets,
            interleave=self.interleave,
            rope_scale=self.rope_scale,
            rope_theta=self.rope_theta,
        )

        # Reshape back to [BS, seq_len, num_heads, head_dim]
        q_rope = q_ragged.reshape(BS_q, seq_len_q, num_heads, head_dim)
        k_rope = k_ragged.reshape(BS_k, seq_len_k, num_heads, head_dim)

        return q_rope, k_rope

    def _apply_rope_single(self, x: torch.Tensor, rope_offset: int = 0) -> torch.Tensor:
        """
        Apply RoPE to a single tensor using FlashInfer's efficient kernel.

        This is useful for cross-attention where Q and K have different batch sizes
        and need to be rotated separately.

        Args:
            x: [BS, seq_len, num_heads, head_dim] - Input tensor
            rope_offset: Position offset for RoPE

        Returns:
            Rotated tensor with same shape
        """
        BS, seq_len, num_heads, head_dim = x.shape

        # Flatten to ragged format: [BS, L, H, D] -> [BS*L, H, D]
        x_ragged = x.reshape(BS * seq_len, num_heads, head_dim)

        # Create ragged tensor indices for RoPE
        indptr = torch.arange(0, (BS + 1) * seq_len, seq_len, dtype=torch.int32, device=x.device)
        # Position offsets for each sequence in the batch
        offsets = torch.full((BS,), rope_offset, dtype=torch.int32, device=x.device)

        # Apply RoPE using flashinfer's efficient kernel
        # When we only have one tensor, we pass it as both q and k, but only use the first output
        x_ragged, _ = flashinfer.rope.apply_rope(  # type: ignore
            x_ragged,
            x_ragged,
            indptr=indptr,
            offsets=offsets,
            interleave=self.interleave,
            rope_scale=self.rope_scale,
            rope_theta=self.rope_theta,
        )

        # Reshape back to [BS, seq_len, num_heads, head_dim]
        x_rope = x_ragged.reshape(BS, seq_len, num_heads, head_dim)

        return x_rope

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        use_rope: bool = False,
        rope_offset: Optional[int] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention using FlashInfer with optional integrated RoPE and KV masking.

        Args:
            q: [BS, seq_len_q, num_heads, head_dim] - Query tensor
            k: [BS, seq_len_k, num_heads, head_dim] - Key tensor
            v: [BS, seq_len_k, num_heads, head_dim] - Value tensor
            causal: Whether to apply causal masking
            use_rope: If True, apply RoPE internally using flashinfer (more efficient)
            rope_offset: Position offset for RoPE (only used when use_rope=True)
            kv_mask: Optional [seq_len_k] boolean mask for KV pairs (True = valid, False = padding)
                     Used for handling variable-length sequences with fixed-shape tensors

        Returns:
            Attention output [BS, seq_len_q, num_heads, head_dim]
        """
        BS, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_kv, _, _ = k.shape

        # Handle KV masking for variable-length sequences
        # During cache warmup (prefill), kv_mask may have False values (padding)
        # During steady-state decode with CUDA Graph, kv_mask should be all-True
        if kv_mask is not None and not kv_mask.all():
            # Mask has padding - compute valid length and slice
            # WARNING: This path is NOT CUDA Graph compatible (dynamic slicing)
            # Only use this during warmup/prefill, not during Graph capture
            valid_kv_len_tensor = kv_mask.sum(dtype=torch.long)
            valid_kv_len = int(valid_kv_len_tensor.item())

            # Truncate k/v to valid length
            k = k[:, :valid_kv_len]
            v = v[:, :valid_kv_len]
            seq_len_kv = valid_kv_len

        if BS == 1:
            # Single sequence - use single_prefill_with_kv_cache
            # Expected shape: [seq_len, num_heads, head_dim]
            q_single = q.squeeze(0)  # [seq_len_q, num_heads, head_dim]
            k_single = k.squeeze(0)  # [seq_len_kv, num_heads, head_dim]
            v_single = v.squeeze(0)  # [seq_len_kv, num_heads, head_dim]

            output = flashinfer.single_prefill_with_kv_cache(
                q_single,
                k_single,
                v_single,
                causal=causal,
                use_fp16_qk_reduction=False,
            )
            # Output shape: [seq_len_q, num_heads, head_dim]
            # Add back batch dimension
            output = output.unsqueeze(0)  # [1, seq_len_q, num_heads, head_dim]
        else:
            # Multiple sequences - use BatchPrefillWithRaggedKVCacheWrapper
            # Flatten to ragged format: [BS, L, H, D] -> [BS*L, H, D]
            total_q_len = BS * seq_len_q
            total_kv_len = BS * seq_len_kv

            q_ragged = q.reshape(total_q_len, num_heads, head_dim)
            k_ragged = k.reshape(total_kv_len, num_heads, head_dim)
            v_ragged = v.reshape(total_kv_len, num_heads, head_dim)

            # Apply RoPE if requested using flashinfer's built-in kernel
            if use_rope and rope_offset is not None:
                # Apply RoPE using flashinfer's efficient kernel
                # This computes cos/sin on-the-fly inside the kernel
                # Note: _apply_rope_internal expects [BS, L, H, D] and returns the same,
                # so we need to reshape before and after
                q_temp = q_ragged.reshape(BS, seq_len_q, num_heads, head_dim)
                k_temp = k_ragged.reshape(BS, seq_len_kv, num_heads, head_dim)
                q_temp, k_temp = self._apply_rope_internal(q_temp, k_temp, rope_offset)
                q_ragged = q_temp.reshape(total_q_len, num_heads, head_dim)
                k_ragged = k_temp.reshape(total_kv_len, num_heads, head_dim)

            # Create ragged tensor indices: qo_indptr and kv_indptr
            # Both [BS+1], marking start/end positions of each sequence
            qo_indptr = torch.arange(
                0, (BS + 1) * seq_len_q, seq_len_q, dtype=torch.int32, device=q.device
            )
            kv_indptr = torch.arange(
                0, (BS + 1) * seq_len_kv, seq_len_kv, dtype=torch.int32, device=k.device
            )

            # Initialize wrapper if needed (allocate workspace buffer)
            if self._batch_wrapper is None or self._workspace_buffer is None:
                # Allocate 128MB workspace buffer (can be adjusted)
                self._workspace_buffer = torch.empty(
                    128 * 1024 * 1024, dtype=torch.uint8, device=q.device
                )
                self._batch_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    self._workspace_buffer, kv_layout="NHD"
                )

            # Plan the attention computation (creates auxiliary data structures)
            self._batch_wrapper.plan(
                qo_indptr,
                kv_indptr,
                num_heads,
                num_heads,  # Assume num_kv_heads == num_qo_heads for now
                head_dim,
                causal=causal,
                q_data_type=q.dtype,  # Use actual dtype of q
                kv_data_type=k.dtype,  # Use actual dtype of k/v
            )

            # Run the attention computation
            output = self._batch_wrapper.run(q_ragged, k_ragged, v_ragged)

            # Reshape back to [BS, seq_len_q, num_heads, head_dim]
            output = output.reshape(BS, seq_len_q, num_heads, head_dim)

        return output
