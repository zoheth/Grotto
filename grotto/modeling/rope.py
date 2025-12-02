import torch


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
