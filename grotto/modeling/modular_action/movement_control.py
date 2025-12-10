from typing import Optional, Tuple

import flashinfer
import torch
from einops import rearrange
from torch import nn

from grotto.modeling.modular_action.action_config import ActionConfig


class MovementPreprocessor(nn.Module):
    def __init__(
        self,
        vae_time_compression_ratio: int,
        windows_size: int,
        movement_dim_in: int,
        hidden_size: int,
    ):
        super().__init__()
        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size
        self.pat_t = vae_time_compression_ratio * windows_size

        self.movement_embed = nn.Sequential(
            nn.Linear(movement_dim_in, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(
        self,
        movement_condition: torch.Tensor,
        is_causal: bool,
        num_frame_per_block: int,
    ) -> torch.Tensor:
        B, N_frames, C = movement_condition.shape

        pad_t = self.pat_t
        pad = movement_condition[:, 0:1, :].expand(-1, pad_t, -1)
        movement_condition_padded = torch.cat([pad, movement_condition], dim=1)

        movement_condition_embedded = self.movement_embed(movement_condition_padded)

        if is_causal:
            N_feats = (N_frames - 1) // self.vae_time_compression_ratio + 1
            start_idx = (
                self.vae_time_compression_ratio
                * (N_feats - num_frame_per_block - self.windows_size)
                + pad_t
            )
            movement_condition_embedded = movement_condition_embedded[:, start_idx:, :]
            group_movement = [
                movement_condition_embedded[
                    :,
                    self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i
                    * self.vae_time_compression_ratio
                    + pad_t,
                    :,
                ]
                for i in range(num_frame_per_block)
            ]
        else:
            N_feats = (N_frames - 1) // self.vae_time_compression_ratio + 1
            group_movement = [
                movement_condition_embedded[
                    :,
                    self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i
                    * self.vae_time_compression_ratio
                    + pad_t,
                    :,
                ]
                for i in range(N_feats)
            ]

        group_movement = torch.stack(group_movement, dim=1)
        group_movement = group_movement.reshape(B, group_movement.shape[1], -1)

        return group_movement


class MovementInjector(nn.Module):
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

        self.preprocessor = MovementPreprocessor(
            action_config.vae_time_compression_ratio,
            action_config.windows_size,
            action_config.keyboard_dim_in,
            action_config.hidden_size,
        )

        self.q_proj = nn.Linear(
            action_config.img_hidden_size,
            action_config.keyboard_hidden_dim,
            bias=action_config.qkv_bias,
        )

        self.kv_proj = nn.Linear(
            action_config.hidden_size
            * action_config.windows_size
            * action_config.vae_time_compression_ratio,
            action_config.keyboard_hidden_dim * 2,
            bias=action_config.qkv_bias,
        )

        head_dim = action_config.keyboard_head_dim
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()

        self.proj_movement = nn.Linear(
            action_config.keyboard_hidden_dim,
            action_config.img_hidden_size,
            bias=action_config.qkv_bias,
        )

        self.num_heads = action_config.heads_num
        self.head_dim = action_config.keyboard_head_dim

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

        movement_features = self.preprocessor(
            condition, is_causal=True, num_frame_per_block=num_frame_per_block
        )

        q = self.q_proj(x)
        q = q.view(B, T, S, self.num_heads, self.head_dim)
        q = q.transpose(1, 2).reshape(B * S, T, self.num_heads, self.head_dim)

        movement_kv = self.kv_proj(movement_features)
        k, v = rearrange(movement_kv, "B L (K H D) -> K B L H D", K=2, H=self.num_heads)

        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        BS, T_q = B * S, T
        B_k, L_k = k.shape[0], k.shape[1]

        # Expand k and v to all spatial locations
        k_expanded = (
            k.view(B_k, 1, L_k, self.num_heads, self.head_dim)
            .expand(-1, S, -1, -1, -1)
            .reshape(BS, L_k, self.num_heads, self.head_dim)
        )
        v_expanded = (
            v.view(B_k, 1, L_k, self.num_heads, self.head_dim)
            .expand(-1, S, -1, -1, -1)
            .reshape(BS, L_k, self.num_heads, self.head_dim)
        )

        # Reshape to [BS*T_q, num_heads, head_dim] and [BS*L_k, num_heads, head_dim]
        q_flat = q.reshape(BS * T_q, self.num_heads, self.head_dim)
        k_flat = k_expanded.reshape(BS * L_k, self.num_heads, self.head_dim)
        v_flat = v_expanded.reshape(BS * L_k, self.num_heads, self.head_dim)

        attn_output = flashinfer.single_prefill_with_kv_cache(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            causal=False,
            kv_layout="NHD",
            pos_encoding_mode="NONE",
        )

        attn_output = attn_output.view(BS, T_q, self.num_heads, self.head_dim)
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_movement(attn_output)

        return x + attn_output
