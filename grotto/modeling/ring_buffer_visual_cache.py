"""Ring Buffer KV Cache with zero-copy eviction and CUDA Graph compatibility."""

from typing import Tuple

import torch


class RingBufferVisualCache:
    """Visual KV cache with O(1) eviction and full CUDA Graph support."""

    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.k_cache = torch.zeros((max_seq_len, num_heads, head_dim), dtype=dtype, device=device)
        self.v_cache = torch.zeros((max_seq_len, num_heads, head_dim), dtype=dtype, device=device)

        self.write_pos = torch.zeros(1, dtype=torch.long, device=device)
        self.valid_len = torch.zeros(1, dtype=torch.long, device=device)
        self.global_end_index = torch.zeros(1, dtype=torch.long, device=device)

        self._arange = torch.arange(max_seq_len, device=device, dtype=torch.long)
        self._max_seq_len_tensor = torch.tensor([max_seq_len], dtype=torch.long, device=device)
        self._zero = torch.zeros(1, dtype=torch.long, device=device)

    def reset(self):
        self.write_pos.zero_()
        self.valid_len.zero_()
        self.global_end_index.zero_()

    def update_or_append(self, k: torch.Tensor, v: torch.Tensor, current_end: int) -> None:
        incoming_len = k.shape[0]
        current_end_tensor = torch.tensor([current_end], dtype=torch.long, device=self.device)
        is_overwrite = current_end_tensor <= self.global_end_index

        if is_overwrite.item():
            write_start_pos = torch.maximum(self._zero, self.write_pos - incoming_len)
            self._write_with_wrap(k, v, write_start_pos, incoming_len)
        else:
            self._write_with_wrap(k, v, self.write_pos, incoming_len)
            self.write_pos.add_(incoming_len).fmod_(self.max_seq_len)
            self.valid_len.add_(incoming_len).clamp_(max=self.max_seq_len)
            self.global_end_index.copy_(current_end_tensor)

    def _write_with_wrap(
        self, k: torch.Tensor, v: torch.Tensor, start_offset: torch.Tensor, incoming_len: int
    ) -> None:
        write_indices = (
            start_offset + torch.arange(incoming_len, device=self.device)
        ) % self.max_seq_len
        self.k_cache[write_indices] = k
        self.v_cache[write_indices] = v

    def evict(self, max_allowed_tokens: int) -> int:
        max_allowed_tensor = torch.tensor(
            [max_allowed_tokens], dtype=torch.long, device=self.device
        )
        tokens_to_remove = torch.maximum(self._zero, self.valid_len - max_allowed_tensor)
        self.valid_len.sub_(tokens_to_remove).clamp_(min=0)
        return int(tokens_to_remove.item())

    def get_kv_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_len_scalar = self.valid_len.squeeze()
        read_indices = (
            self.write_pos - valid_len_scalar + self._arange[: self.max_seq_len]
        ) % self.max_seq_len
        valid_indices = read_indices[:valid_len_scalar]
        return self.k_cache[valid_indices], self.v_cache[valid_indices]

    @property
    def total_tokens(self) -> int:
        return int(self.valid_len.item())

    def __repr__(self) -> str:
        return (
            f"RingBufferVisualCache("
            f"valid_len={self.valid_len.item()}, "
            f"write_pos={self.write_pos.item()}, "
            f"max_seq_len={self.max_seq_len}, "
            f"heads={self.num_heads}, "
            f"head_dim={self.head_dim})"
        )
