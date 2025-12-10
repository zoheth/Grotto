from typing import Optional, Tuple

import flashinfer
import torch
from einops import rearrange
from torch import nn

from grotto.modeling.modular_action.action_config import ActionConfig
from grotto.modeling.modular_action.interfaces import ActionInjector


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


class MovementInjector(ActionInjector):
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

        if workspace_buffer is None:
            workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        self.workspace_buffer = workspace_buffer
        self.flashinfer_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD"
        )

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

        rope_theta = getattr(self.action_config, "rope_theta", 256.0)

        BS, T_q = B * S, T
        q_ragged = q.reshape(BS * T_q, self.num_heads, self.head_dim)
        indptr_q = torch.arange(0, BS + 1, dtype=torch.int32, device=q.device) * T_q
        offsets_q = torch.full((BS,), start_frame, dtype=torch.int32, device=q.device)
        roped_q, _ = flashinfer.apply_rope(
            q_ragged, q_ragged, indptr_q, offsets_q, interleave=False, rope_theta=rope_theta
        )

        B_k, L_k = k.shape[0], k.shape[1]
        k_ragged = k.reshape(B_k * L_k, self.num_heads, self.head_dim)
        indptr_k = torch.arange(0, B_k + 1, dtype=torch.int32, device=k.device) * L_k
        offsets_k = torch.full((B_k,), start_frame, dtype=torch.int32, device=k.device)
        _, roped_k = flashinfer.apply_rope(
            k_ragged, k_ragged, indptr_k, offsets_k, interleave=False, rope_theta=rope_theta
        )

        roped_k_expanded = (
            roped_k.view(B_k, 1, L_k, self.num_heads, self.head_dim)
            .expand(-1, S, -1, -1, -1)
            .reshape(BS * L_k, self.num_heads, self.head_dim)
        )
        v_expanded = (
            v.reshape(B_k, 1, L_k, self.num_heads, self.head_dim)
            .expand(-1, S, -1, -1, -1)
            .reshape(BS * L_k, self.num_heads, self.head_dim)
        )

        qo_indptr = torch.arange(0, BS + 1, dtype=torch.int32, device=q.device) * T_q
        kv_indptr = torch.arange(0, BS + 1, dtype=torch.int32, device=q.device) * L_k

        self.flashinfer_wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            q_data_type=roped_q.dtype,
        )

        attn_output = self.flashinfer_wrapper.run(roped_q, roped_k_expanded, v_expanded)

        attn_output = attn_output.view(BS, T_q, self.num_heads, self.head_dim)
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_movement(attn_output)

        return x + attn_output
