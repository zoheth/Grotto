"""
View Control Module - Handles camera/view angle conditioning.

This module implements view control injection (originally mouse-based input).
It uses self-attention with RoPE to incorporate view angle changes into the model.

Terminology:
    View Control = Camera control = Look direction
    Input sources: Mouse movement, gamepad right stick, etc.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from grotto.modeling.modular_action.action_config import ActionConfig
from grotto.modeling.modular_action.interfaces import ActionInjector, AttentionKernel
from grotto.modeling.modular_action.kernels.preprocessor_kernel import mouse_preprocessor_triton

if TYPE_CHECKING:
    from ..ring_buffer_cache import RingBufferActionCache


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

    def __init__(self, action_config: ActionConfig):
        super().__init__()
        self.action_config = action_config

        # Preprocessor
        self.preprocessor = ViewControlPreprocessor(
            action_config.vae_time_compression_ratio, action_config.windows_size
        )

        # MLP to fuse hidden states with view control condition
        view_input_dim = (
            action_config.img_hidden_size
            + action_config.mouse_dim_in  # Config still uses "mouse" naming
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
        self.t_qkv = nn.Linear(
            action_config.mouse_hidden_dim,
            action_config.mouse_hidden_dim * 3,
            bias=action_config.qkv_bias,
        )

        # QK normalization
        head_dim = action_config.mouse_head_dim
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()

        # Output projection
        self.proj_view = nn.Linear(
            action_config.mouse_hidden_dim,
            action_config.img_hidden_size,
            bias=action_config.qkv_bias,
        )

        # Attention core
        self.attn_core = AttentionKernel()

        # Cache configuration
        self.max_attention_size = action_config.local_attn_size

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        is_causal: bool = False,
        kv_cache: Optional["RingBufferActionCache"] = None,
        start_frame: int = 0,
        num_frame_per_block: int = 1,
    ) -> torch.Tensor:
        """
        Forward pass for view control condition injection.

        Args:
            x: [B, T*S, C_img] - Input hidden states
            condition: [B, N_frames, C_view] - View control condition
            spatial_shape: (H, W) spatial dimensions
            temporal_shape: T temporal dimension
            is_causal: Whether to use causal attention
            kv_cache: KV cache for incremental decoding
            start_frame: Starting frame index for RoPE
            num_frame_per_block: Number of frames per block

        Returns:
            Output hidden states [B, T*S, C_img]
        """
        if condition is None:
            return x

        B, T_S, C_img = x.shape
        T = temporal_shape
        H, W = spatial_shape
        S = H * W

        # Fuse with view control condition using triton kernel (accepts B, T*S, C directly)
        # Output: [B, S, T, C_fused]
        fused_features = mouse_preprocessor_triton(
            x,  # [B, T*S, C_img]
            condition,
            T,  # temporal_shape
            self.action_config.vae_time_compression_ratio,
            self.action_config.windows_size,
            is_causal,
            num_frame_per_block,
        )

        # Merge B and S dimensions: [B, S, T, C_fused] -> [B*S, T, C_fused]
        fused_features = fused_features.reshape(B * S, T, -1)

        # MLP
        fused_features = self.view_mlp(fused_features)

        # QKV projection and split
        qkv = self.t_qkv(fused_features)  # [BS, T, 3*C_view]
        q, k, v = rearrange(qkv, "BS T (K H D) -> K BS T H D", K=3, H=self.action_config.heads_num)

        # QK normalization
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        # Attention computation with integrated RoPE
        # Using FlashInfer's built-in RoPE is more efficient - it computes cos/sin on-the-fly
        # and avoids separate kernel launches and memory transfers
        if is_causal and kv_cache is not None:
            # For causal mode with KV cache, we need to:
            # 1. Apply RoPE to new Q/K using FlashInfer's integrated kernel
            # 2. Update cache with the rotated K/V
            # 3. Get the attention window from cache
            # 4. Compute attention

            # Apply RoPE using FlashInfer's integrated kernel (more efficient than external apply_rotary_emb)
            q_rope, k_rope = self.attn_core._apply_rope_internal(q, k, start_frame)

            # Update KV cache and get window directly
            k_window, v_window, kv_mask = kv_cache.update_and_get_window(
                k=k_rope,
                v=v,
                num_new_tokens=num_frame_per_block,
                max_attention_size=self.max_attention_size,
            )
            # Compute attention with cached KV (RoPE already applied, pass mask for padding handling)
            attn_output = self.attn_core(
                q_rope, k_window, v_window, causal=False, use_rope=False, kv_mask=kv_mask
            )
        else:
            # Regular attention: use FlashInfer's integrated RoPE for best performance
            attn_output = self.attn_core(
                q, k, v, causal=is_causal, use_rope=True, rope_offset=start_frame
            )

        # Reshape and project: [BS, T, H, D] -> [B, T*S, C_img]
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_view(attn_output)

        # Residual connection
        output = x + attn_output
        return output
