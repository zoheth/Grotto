from typing import List

import torch


class RoPE3DCache:
    def __init__(
        self,
        freqs: torch.Tensor,
        height: int,
        width: int,
        max_frames: int = 150,
        rope_dim_list: List[int] | None = None,
    ):
        if rope_dim_list is None:
            rope_dim_list = [8, 28, 28]
        device = freqs.device
        dtype = torch.float32

        # Fix: correctly handle complex input (cis) by extracting phase angles
        if freqs.is_complex():
            freqs = freqs.angle().to(device=device, dtype=dtype)
        else:
            freqs = freqs.to(device=device, dtype=dtype)

        total_rope_dim = sum(rope_dim_list)
        freqs_dim = freqs.shape[1]

        if total_rope_dim == freqs_dim * 2:
            d_time, d_height, d_width = (d // 2 for d in rope_dim_list)
        elif total_rope_dim == freqs_dim:
            d_time, d_height, d_width = rope_dim_list
        else:
            raise ValueError(
                f"rope_dim_list sum {total_rope_dim} mismatch with freqs dim {freqs_dim}"
            )

        freqs_time = freqs[:max_frames, :d_time]
        freqs_height = freqs[:height, d_time : d_time + d_height]
        freqs_width = freqs[:width, d_time + d_height :]

        # 1. 3D Global Cache
        t_grid = freqs_time.view(max_frames, 1, 1, d_time).expand(max_frames, height, width, d_time)
        h_grid = freqs_height.view(1, height, 1, d_height).expand(
            max_frames, height, width, d_height
        )
        w_grid = freqs_width.view(1, 1, width, d_width).expand(max_frames, height, width, d_width)

        freqs_3d = torch.cat([t_grid, h_grid, w_grid], dim=-1).reshape(-1, freqs_dim)

        self.cache_3d = torch.cat([torch.cos(freqs_3d), torch.sin(freqs_3d)], dim=-1).contiguous()

    def get_cache(self) -> torch.Tensor:
        return self.cache_3d
