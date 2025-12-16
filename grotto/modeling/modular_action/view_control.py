from typing import Optional, Tuple

import flashinfer
import torch
from einops import rearrange
from torch import nn

from grotto.modeling.modular_action.action_config import ActionConfig


class ViewControlInjector(nn.Module):
    def __init__(
        self,
        action_config: ActionConfig,
        num_frame_per_block: int,
        height: int,
        width: int,
        workspace_buffer: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.action_config = action_config
        self.num_frame_per_block = num_frame_per_block
        self.height = height
        self.width = width
        self._workspace_buffer = workspace_buffer

        view_input_dim = (
            action_config.img_hidden_size
            + action_config.mouse_dim_in
            * action_config.vae_time_compression_ratio
            * action_config.windows_size
        )

        self.view_mlp = nn.Sequential(
            nn.Linear(view_input_dim, action_config.mouse_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_config.mouse_hidden_dim, action_config.mouse_hidden_dim),
            nn.LayerNorm(action_config.mouse_hidden_dim),
        )

        self.num_heads = action_config.heads_num
        self.head_dim = action_config.mouse_head_dim

        self.t_qkv = nn.Linear(
            action_config.mouse_hidden_dim,
            action_config.mouse_hidden_dim * 3,
            bias=action_config.qkv_bias,
        )

        self.q_norm = (
            nn.RMSNorm(self.head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.RMSNorm(self.head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()
        )

        self.proj_view = nn.Linear(
            action_config.mouse_hidden_dim,
            action_config.img_hidden_size,
            bias=action_config.qkv_bias,
        )

        # No longer need workspace buffer and wrapper for single_prefill_with_kv_cache

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        num_frame_per_block: int = 1,
    ) -> torch.Tensor:
        if condition is None:
            return x

        B, T_S, C_img = x.shape
        T = temporal_shape
        H, W = spatial_shape
        S = H * W

        if condition.shape[1] > 9:
            condition = condition[:, -12:, :]

        B, N_frames, C = condition.shape  # (1, 9, 2) (1, 12, 2)
        action_len = 20
        pad_len = 12
        zeros_pad = torch.zeros(
            (B, action_len - N_frames, C), device=condition.device, dtype=condition.dtype
        )
        mouse_padded = torch.cat([zeros_pad, condition], dim=1)
        windows = []
        for i in range(num_frame_per_block):  # 0, 1, 2
            start = i * 4  # ratio * i
            end = start + pad_len
            win = mouse_padded[:, start:end, :]
            windows.append(win)

        group_mouse = torch.stack(windows, dim=1)  # (1, 3, 12, 2)

        # ==========================================
        # 3. 维度调整 (Flatten & Expand)
        # ==========================================
        # 变为 (1, 3, 24)
        group_mouse = group_mouse.flatten(2)

        # 广播到空间维度 S
        S = 880  # th * tw
        group_mouse = group_mouse.unsqueeze(1).expand(-1, S, -1, -1)  # (1, 880, 3, 24)
        group_mouse = group_mouse.reshape(B * S, num_frame_per_block, -1)  # (880, 3, 24)
        fused = torch.cat([x.reshape(B * S, T, C_img), group_mouse], dim=-1)

        # fused = rotation_preprocessor_triton(
        #     x,
        #     condition,
        #     T,
        #     self.action_config.vae_time_compression_ratio,
        #     self.action_config.windows_size,
        #     num_frame_per_block,
        # )
        # fused = fused.reshape(B * S, T, -1)
        fused = self.view_mlp(fused)

        qkv = self.t_qkv(fused)
        q, k, v = rearrange(qkv, "BS T (three H D) -> three BS T H D", three=3, H=self.num_heads)
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        BS = B * S

        q_flat = q.reshape(BS * T, self.num_heads, self.head_dim)
        k_flat = k.reshape(BS * T, self.num_heads, self.head_dim)
        v_flat = v.reshape(BS * T, self.num_heads, self.head_dim)

        attn_output = flashinfer.single_prefill_with_kv_cache(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            causal=False,
            kv_layout="NHD",
            pos_encoding_mode="NONE",
        )

        attn_output = attn_output.view(BS, T, self.num_heads, self.head_dim)
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_view(attn_output)

        return x + attn_output
