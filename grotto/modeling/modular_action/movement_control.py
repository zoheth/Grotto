"""
Movement Control Module - Handles character movement conditioning.

This module implements movement control injection (originally keyboard-based input).
It uses cross-attention with RoPE to incorporate movement commands into the model.

Terminology:
    Movement Control = Character movement = Position change
    Input sources: Keyboard (WASD), gamepad left stick, etc.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from grotto.modeling.modular_action.action_config import ActionConfig
from grotto.modeling.modular_action.interfaces import ActionInjector, AttentionKernel

if TYPE_CHECKING:
    from ..ring_buffer_cache import RingBufferActionCache


class MovementPreprocessor(nn.Module):
    """
    Preprocessor for movement control condition data.

    Processes discrete movement commands (e.g., key presses) into continuous embeddings
    using a sliding window to capture temporal patterns.
    """

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

        # Movement embedding layers - maps discrete movement input to continuous space
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
        """
        Process movement condition into windowed features.

        Args:
            movement_condition: [B, N_frames, C_movement] - Raw movement condition
            is_causal: Whether in causal mode
            num_frame_per_block: Number of frames per block in causal mode

        Returns:
            Windowed movement features [B, T_k, C_movement * windows]
        """
        B, N_frames, C = movement_condition.shape

        # Padding
        pad_t = self.pat_t
        pad = movement_condition[:, 0:1, :].expand(-1, pad_t, -1)
        movement_condition_padded = torch.cat([pad, movement_condition], dim=1)

        # Embed
        movement_condition_embedded = self.movement_embed(movement_condition_padded)

        # Extract windows
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

        # Stack and flatten: [B, T_k, windows * C]
        group_movement = torch.stack(group_movement, dim=1)
        group_movement = group_movement.reshape(B, group_movement.shape[1], -1)

        return group_movement


class MovementInjector(ActionInjector):
    """
    Movement control condition injector using cross-attention with RoPE.

    This module injects character movement signals (e.g., from keyboard WASD or gamepad left stick)
    into the model's hidden states. It uses cross-attention where queries come from visual features
    and keys/values come from movement commands.

    Architecture:
        1. Preprocess: Embed movement condition (sliding window)
        2. Q Projection: Project hidden states to query space
        3. KV Projection: Project movement condition to key-value space
        4. Cross-Attention: Q from visual, KV from movement, with RoPE
        5. Output: Project back to model hidden size with residual connection
    """

    def __init__(self, action_config: ActionConfig):
        super().__init__()
        self.action_config = action_config

        # Preprocessor
        self.preprocessor = MovementPreprocessor(
            action_config.vae_time_compression_ratio,
            action_config.windows_size,
            action_config.keyboard_dim_in,  # Config still uses "keyboard" naming
            action_config.hidden_size,
        )

        # Query projection (from hidden states)
        self.q_proj = nn.Linear(
            action_config.img_hidden_size,
            action_config.keyboard_hidden_dim,
            bias=action_config.qkv_bias,
        )

        # Key-Value projection (from movement condition)
        self.kv_proj = nn.Linear(
            action_config.hidden_size
            * action_config.windows_size
            * action_config.vae_time_compression_ratio,
            action_config.keyboard_hidden_dim * 2,
            bias=action_config.qkv_bias,
        )

        # QK normalization
        head_dim = action_config.keyboard_head_dim
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()

        # Output projection
        self.proj_movement = nn.Linear(
            action_config.keyboard_hidden_dim,
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
        Forward pass for movement control condition injection.

        Args:
            x: [B, T*S, C_img] - Input hidden states
            condition: [B, N_frames, C_movement] - Movement control condition
            spatial_shape: (H, W) spatial dimensions
            temporal_shape: T temporal dimension
            is_causal: Whether to use causal attention
            kv_cache: RingBufferActionCache for incremental decoding
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

        # Process movement condition
        group_movement = self.preprocessor(condition, is_causal, num_frame_per_block)

        # Compute Query from hidden states
        q = self.q_proj(x)  # [B, T*S, C_movement]
        q = q.view(B, T_S, self.action_config.heads_num, self.action_config.keyboard_head_dim)

        # Compute Key-Value from movement condition
        movement_kv = self.kv_proj(group_movement)  # [B, T_k, 2*C_movement]
        k, v = rearrange(
            movement_kv,
            "B T (K H D) -> K B T H D",
            K=2,
            H=self.action_config.heads_num,
            D=self.action_config.keyboard_head_dim,
        )

        # QK normalization
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        # Reshape Q for cross-attention: [B, T*S, H, D] -> [B*S, T, H, D]
        q = rearrange(q, "B (T S) H D -> (B S) T H D", T=T, S=S)

        # Attention computation with integrated RoPE
        # For cross-attention with different batch sizes, apply RoPE separately to Q and K
        # This is most efficient: each tensor only needs one RoPE kernel call

        # Apply RoPE to Q: [B*S, T, H, D] -> [B*S, T, H, D]
        q_rope = self.attn_core._apply_rope_single(q, start_frame)

        # Apply RoPE to K: [B, T_k, H, D] -> [B, T_k, H, D]
        k_rope = self.attn_core._apply_rope_single(k, start_frame)

        # Extract dimensions for cache operations
        T_k = k_rope.shape[1]  # Temporal dimension of movement condition
        num_heads = self.action_config.heads_num
        head_dim = self.action_config.keyboard_head_dim

        if is_causal and kv_cache is not None:
            # Use standard PagedCache path (Triton kernel can be re-implemented later)
            # Need to mean-pool spatially before caching: [B, T_k, H, D] -> [B, 1, H, D] for view
            # But for now, we cache per-spatial-location

            # For view cache, we cache [B*S, T_k, H, D] format
            # Expand to spatial dimension: [B, T_k, H, D] -> [B, S, T_k, H, D]
            k_rope_expanded = k_rope.unsqueeze(1).expand(-1, S, -1, -1, -1)
            v_expanded = v.unsqueeze(1).expand(-1, S, -1, -1, -1)

            # Reshape to [B*S, T_k, H, D] for cache update
            k_rope_for_cache = k_rope_expanded.reshape(B * S, T_k, num_heads, head_dim)
            v_for_cache = v_expanded.reshape(B * S, T_k, num_heads, head_dim)

            # Update KV cache directly
            # Note: Currently assumes B*S = 1, so we take [0] slice
            k_window, v_window, kv_mask = kv_cache.update_and_get_window(
                k=k_rope_for_cache[0:1],  # [1, T_k, H, D]
                v=v_for_cache[0:1],  # [1, T_k, H, D]
                num_new_tokens=T_k,
                max_attention_size=self.max_attention_size,
            )

            # Expand window to all spatial locations: [1, window_len, H, D] -> [B*S, window_len, H, D]
            k_window = k_window.expand(B * S, -1, -1, -1)
            v_window = v_window.expand(B * S, -1, -1, -1)

            # Compute attention with cached KV (RoPE already applied, pass mask for padding handling)
            attn_output = self.attn_core(
                q_rope, k_window, v_window, causal=False, use_rope=False, kv_mask=kv_mask
            )
        else:
            # Regular cross-attention
            # Expand K, V to match spatial dimension: [B, T_k, H, D] -> [B*S, T_k, H, D]
            k_rope_expanded = (
                k_rope.unsqueeze(1)
                .expand(-1, S, -1, -1, -1)
                .reshape(B * S, -1, k_rope.shape[-2], k_rope.shape[-1])
            )
            v_expanded = (
                v.unsqueeze(1)
                .expand(-1, S, -1, -1, -1)
                .reshape(B * S, -1, v.shape[-2], v.shape[-1])
            )

            # Compute cross-attention (RoPE already applied)
            attn_output = self.attn_core(
                q_rope, k_rope_expanded, v_expanded, causal=False, use_rope=False
            )

        # Reshape and project: [B*S, T, H, D] -> [B, T*S, C_img]
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_movement(attn_output)

        # Residual connection
        output = x + attn_output
        return output
