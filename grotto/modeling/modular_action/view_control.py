from typing import Optional, Tuple

import flashinfer
import torch
from einops import rearrange
from torch import nn

from grotto.modeling.modular_action.action_config import ActionConfig
from grotto.modeling.modular_action.interfaces import ActionInjector
from grotto.modeling.modular_action.kernels.preprocessor_kernel import rotation_preprocessor_triton


class ViewControlInjector(ActionInjector):
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

        fused = rotation_preprocessor_triton(
            x,
            condition,
            T,
            self.action_config.vae_time_compression_ratio,
            self.action_config.windows_size,
            num_frame_per_block,
        )
        fused = fused.reshape(B * S, T, -1)
        fused = self.view_mlp(fused)

        qkv = self.t_qkv(fused)
        q, k, v = rearrange(qkv, "BS T (three H D) -> three BS T H D", three=3, H=self.num_heads)
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        BS = B * S
        q_ragged = q.reshape(BS * T, self.num_heads, self.head_dim)
        k_ragged = k.reshape(BS * T, self.num_heads, self.head_dim)

        indptr = torch.arange(0, BS + 1, dtype=torch.int32, device=q.device) * T
        offsets = torch.full((BS,), start_frame, dtype=torch.int32, device=q.device)

        rope_theta = getattr(self.action_config, "rope_theta", 10000.0)
        roped_q, roped_k = flashinfer.apply_rope(
            q_ragged, k_ragged, indptr, offsets, interleave=False, rope_theta=rope_theta
        )

        v_ragged = v.reshape(BS * T, self.num_heads, self.head_dim)

        qo_indptr = torch.tensor([0, BS * T], dtype=torch.int32, device=q.device)
        kv_indptr = torch.tensor([0, BS * T], dtype=torch.int32, device=q.device)

        self.flashinfer_wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            q_data_type=roped_q.dtype,
        )

        attn_output = self.flashinfer_wrapper.run(roped_q, roped_k, v_ragged)

        attn_output = attn_output.view(BS, T, self.num_heads, self.head_dim)
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_view(attn_output)

        return x + attn_output
