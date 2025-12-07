"""
View Control Module - Handles camera/view angle conditioning.

This module implements view control injection (originally mouse-based input).
It uses self-attention with RoPE to incorporate view angle changes into the model.

Terminology:
    View Control = Camera control = Look direction
    Input sources: Mouse movement, gamepad right stick, etc.
"""

from typing import Optional, Tuple

import flashinfer
import torch
from einops import rearrange
from torch import nn

from grotto.modeling.attention import AttentionWithCache
from grotto.modeling.kv_cache import DualPlaneKVCache
from grotto.modeling.modular_action.action_config import ActionConfig
from grotto.modeling.modular_action.interfaces import ActionInjector
from grotto.modeling.modular_action.kernels.preprocessor_kernel import rotation_preprocessor_triton


class ViewControlPreprocessor(nn.Module):
    """
    Preprocessor for view control condition data.

    Fuses view control data (camera movement) with hidden states using sliding window.
    This captures local patterns in camera motion trajectory.
    """

    def __init__(self, vae_time_compression_ratio: int, windows_size: int):
        super().__init__()
        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size
        self.pat_t = vae_time_compression_ratio * windows_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        view_control_condition: torch.Tensor,
        is_causal: bool,
        num_frame_per_block: int,
    ) -> torch.Tensor:
        """
        Fuse hidden states with view control condition using sliding window.

        Args:
            hidden_states: [BS, T_q, C_hidden] - Reshaped hidden states
            view_control_condition: [B, N_frames, C_view] - Raw view control condition
            is_causal: Whether in causal mode
            num_frame_per_block: Number of frames per block in causal mode

        Returns:
            Fused features [BS, T_q, C_hidden + C_view * windows]
        """
        B, N_frames, C_view = view_control_condition.shape
        BS, T_q, C_hidden = hidden_states.shape

        # Padding for sliding window
        pad_t = self.pat_t  # vae_time_compression_ratio * windows_size
        pad = view_control_condition[:, 0:1, :].expand(-1, pad_t, -1)
        view_control_padded = torch.cat([pad, view_control_condition], dim=1)

        # Extract windows
        if is_causal:
            N_feats = (N_frames - 1) // self.vae_time_compression_ratio + 1
            # Start from proper position for causal mode
            start_global_idx = N_feats - num_frame_per_block
            end_global_idx = N_feats

            group_view = [
                view_control_padded[
                    :,
                    self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i
                    * self.vae_time_compression_ratio
                    + pad_t,
                    :,
                ]
                for i in range(start_global_idx, end_global_idx)
            ]
        else:
            N_feats = T_q  # Should match temporal shape
            group_view = [
                view_control_padded[
                    :,
                    self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i
                    * self.vae_time_compression_ratio
                    + pad_t,
                    :,
                ]
                for i in range(N_feats)
            ]

        # Stack and expand: [B, T_q, pad_t, C_view]
        group_view = torch.stack(group_view, dim=1)

        # Expand to match spatial dimension: [B, T_q, pad_t, C_view] -> [BS, T_q, pad_t * C_view]
        S = BS // B
        group_view = group_view.unsqueeze(2).expand(B, -1, S, pad_t, C_view)
        group_view = rearrange(group_view, "B T S W C -> (B S) T (W C)")

        # Concatenate with hidden states
        fused = torch.cat([hidden_states, group_view], dim=-1)
        return fused


class ViewControlInjector(ActionInjector):
    """
    View control condition injector using self-attention with RoPE.

    This module injects camera/view control signals (e.g., from mouse or gamepad right stick)
    into the model's hidden states. It uses self-attention to capture correlations between
    view changes and visual content.

    Architecture:
        1. Preprocess: Fuse hidden states with view control condition (sliding window)
        2. MLP: Project fused features to attention hidden dim
        3. Self-Attention: Q/K/V all from fused features, with RoPE
        4. Output: Project back to model hidden size with residual connection
    """

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
        self.frame_seq_len = height * width
        self.block_seq_len = self.frame_seq_len * num_frame_per_block

        # Preprocessor
        self.preprocessor = ViewControlPreprocessor(
            action_config.vae_time_compression_ratio, action_config.windows_size
        )

        # MLP to fuse hidden states with view control condition
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

        # QKV projection
        self.num_heads = action_config.heads_num
        self.head_dim = action_config.mouse_head_dim
        self.t_qkv = nn.Linear(
            action_config.mouse_hidden_dim,
            action_config.mouse_hidden_dim * 3,
            bias=action_config.qkv_bias,
        )

        # QK normalization
        self.q_norm = (
            nn.RMSNorm(self.head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.RMSNorm(self.head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()
        )

        # Output projection
        self.proj_view = nn.Linear(
            action_config.mouse_hidden_dim,
            action_config.img_hidden_size,
            bias=action_config.qkv_bias,
        )

        if workspace_buffer is None:
            workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        self.kv_indptr = torch.zeros(2, dtype=torch.int32)
        self.qo_indptr = torch.zeros(2, dtype=torch.int32)
        self.flashinfer_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD"
        )

        # Cache configuration
        self.max_attention_size = action_config.local_attn_size

        self.register_buffer(
            "local_indices", torch.arange(self.block_seq_len, dtype=torch.int32), persistent=False
        )

        self.attn_backend = AttentionWithCache(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_frame_per_block=self.num_frame_per_block,
            block_seq_len=self.block_seq_len,
            local_attn_size=action_config.local_attn_size,
            workspace_buffer=workspace_buffer,
        )

    def plan_kv_and_attention(
        self,
        incoming_len: int,
        kv_cache: DualPlaneKVCache,
        current_start: int,
        current_end: int,
        grid_sizes: Tuple[int, int, int],
        cache_mode: str = "read_write",
    ) -> None:
        _, height, width = grid_sizes
        self.attn_backend.plan(
            incoming_len=incoming_len,
            kv_cache=kv_cache,
            current_start=current_start,
            current_end=current_end,
            frame_seqlen=height * width,
            cache_mode=cache_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        freqs: torch.Tensor,
        kv_cache: Optional["DualPlaneKVCache"],
        start_frame: int = 0,
        num_frame_per_block: int = 3,
        cache_mode: str = "read_write",
    ) -> torch.Tensor:
        """
        Forward pass for view control condition injection.

        Args:
            x: [B, T*S, C_img] - Input hidden states
            condition: [B, N_frames, C_view] - View control condition
            spatial_shape: (H, W) spatial dimensions
            temporal_shape: T temporal dimension
            freqs_cos: RoPE cos frequencies
            freqs_sin: RoPE sin frequencies
            kv_cache: KV cache for incremental decoding
            start_frame: Starting frame index for RoPE
            num_frame_per_block: Number of frames per block
            cache_mode: "read_write" or "read_only"

        Returns:
            Output hidden states [B, T*S, C_img]
        """
        if condition is None:
            return x

        B, T_S, C_img = x.shape
        T = temporal_shape
        H, W = spatial_shape
        S = H * W

        # [B, T*S, C_img] -> [B, S, T, C_fused]
        fused = rotation_preprocessor_triton(
            x,
            condition,
            T,
            self.action_config.vae_time_compression_ratio,
            self.action_config.windows_size,
            num_frame_per_block,
        )
        # [B, S, T, C_fused] -> [B*S, T, C_fused]
        fused = fused.reshape(B * S, T, -1)
        fused = self.view_mlp(fused)

        # [B*S, T, C] -> [B*S, T, 3*H*D] -> [B*S, T, H, D]
        qkv = self.t_qkv(fused)
        q, k, v = rearrange(qkv, "BS T (three H D) -> three BS T H D", three=3, H=self.num_heads)
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply 1D temporal RoPE: BS independent sequences, each of length T
        BS, T, H, D = q.shape
        q_ragged = q.reshape(BS * T, H, D)  # [BS*T, H, D]
        k_ragged = k.reshape(BS * T, H, D)

        indptr = torch.arange(0, BS + 1, dtype=torch.int32, device=q.device) * T
        offsets = torch.full((BS,), start_frame, dtype=torch.int32, device=q.device)

        rope_theta = getattr(self.action_config, "rope_theta", 256.0)
        roped_q, roped_k = flashinfer.apply_rope(
            q_ragged, k_ragged, indptr, offsets, interleave=False, rope_theta=rope_theta
        )

        v = v.reshape(BS * T, H, D)

        attn_output = self.attn_backend(
            roped_q=roped_q, roped_k=roped_k, v=v, kv_cache=kv_cache, cache_mode=cache_mode
        )

        # [BS*T, H, D] -> [BS, T, H, D] -> [B, T*S, H*D]
        attn_output = attn_output.view(BS, T, H, D)
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_view(attn_output)

        return x + attn_output
