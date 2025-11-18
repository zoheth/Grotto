from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from einops import rearrange

from .interfaces import (
    IAttentionInjector,
    IActionPreprocessor,
    KVCacheManager,
    FlashInferAttentionCore,
    WanRMSNorm,
)
from .action_config import ActionConfig

from .kernels.preprocessor_kernel import mouse_preprocessor_triton
from .kernels.before_attn_kernel import update_kv_cache_triton

class MousePreprocessor(IActionPreprocessor):
    """
    Preprocessor for mouse condition data.
    Fuses mouse condition with hidden states using sliding window.
    """

    def __init__(self, vae_time_compression_ratio: int, windows_size: int):
        super().__init__(vae_time_compression_ratio, windows_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mouse_condition: torch.Tensor,
        is_causal: bool,
        num_frame_per_block: int,
    ) -> torch.Tensor:
        """
        Fuse hidden states with mouse condition using sliding window.

        Args:
            hidden_states: [BS, T_q, C_hidden] - Reshaped hidden states
            mouse_condition: [B, N_frames, C_mouse] - Raw mouse condition
            is_causal: Whether in causal mode
            num_frame_per_block: Number of frames per block in causal mode

        Returns:
            Fused features [BS, T_q, C_hidden + C_mouse * windows]
        """
        B, N_frames, C_mouse = mouse_condition.shape
        BS, T_q, C_hidden = hidden_states.shape

        # Padding for sliding window
        pad_t = self.pat_t  # vae_time_compression_ratio * windows_size
        pad = mouse_condition[:, 0:1, :].expand(-1, pad_t, -1)
        mouse_condition_padded = torch.cat([pad, mouse_condition], dim=1)

        # Extract windows
        if is_causal:
            N_feats = (N_frames - 1) // self.vae_time_compression_ratio + 1
            # Start from proper position for causal mode
            start_global_idx = N_feats - num_frame_per_block
            end_global_idx = N_feats

            group_mouse = [
                mouse_condition_padded[
                    :, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :
                ]
                for i in range(start_global_idx, end_global_idx)
            ]
        else:
            N_feats = T_q  # Should match temporal shape
            group_mouse = [
                mouse_condition_padded[
                    :, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :
                ]
                for i in range(N_feats)
            ]

        # Stack and expand: [B, T_q, pad_t, C_mouse]
        group_mouse = torch.stack(group_mouse, dim=1)

        # Expand to match spatial dimension: [B, T_q, pad_t, C_mouse] -> [BS, T_q, pad_t * C_mouse]
        S = BS // B
        group_mouse = group_mouse.unsqueeze(2).expand(B, -1, S, pad_t, C_mouse)
        group_mouse = rearrange(group_mouse, "B T S W C -> (B S) T (W C)")

        # Concatenate with hidden states
        fused = torch.cat([hidden_states, group_mouse], dim=-1)
        return fused


class KeyboardPreprocessor(IActionPreprocessor):
    """Preprocessor for keyboard condition data."""

    def __init__(self, vae_time_compression_ratio: int, windows_size: int,
                 keyboard_dim_in: int, hidden_size: int):
        super().__init__(vae_time_compression_ratio, windows_size)
        # Keyboard embedding layers - 将 keyboard 输入映射到 hidden_size
        self.keyboard_embed = nn.Sequential(
            nn.Linear(keyboard_dim_in, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(
        self,
        keyboard_condition: torch.Tensor,
        is_causal: bool,
        num_frame_per_block: int,
    ) -> torch.Tensor:
        """
        Process keyboard condition into windowed features.

        Args:
            keyboard_condition: [B, N_frames, C_keyboard] - Raw keyboard condition
            is_causal: Whether in causal mode
            num_frame_per_block: Number of frames per block in causal mode

        Returns:
            Windowed keyboard features [B, T_k, C_keyboard * windows]
        """
        B, N_frames, C = keyboard_condition.shape

        # Padding
        pad_t = self.pat_t
        pad = keyboard_condition[:, 0:1, :].expand(-1, pad_t, -1)
        keyboard_condition_padded = torch.cat([pad, keyboard_condition], dim=1)

        # Embed
        keyboard_condition_embedded = self.keyboard_embed(keyboard_condition_padded)

        # Extract windows
        if is_causal:
            N_feats = (N_frames - 1) // self.vae_time_compression_ratio + 1
            start_idx = self.vae_time_compression_ratio * (N_feats - num_frame_per_block - self.windows_size) + pad_t
            keyboard_condition_embedded = keyboard_condition_embedded[:, start_idx:, :]
            group_keyboard = [
                keyboard_condition_embedded[
                    :, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :
                ]
                for i in range(num_frame_per_block)
            ]
        else:
            N_feats = (N_frames - 1) // self.vae_time_compression_ratio + 1
            group_keyboard = [
                keyboard_condition_embedded[
                    :, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :
                ]
                for i in range(N_feats)
            ]

        # Stack and flatten: [B, T_k, windows * C]
        group_keyboard = torch.stack(group_keyboard, dim=1)
        group_keyboard = group_keyboard.reshape(B, group_keyboard.shape[1], -1)

        return group_keyboard


class MouseInjector(IAttentionInjector):
    """
    Mouse condition injector using self-attention with RoPE.
    Based on the original ActionModule mouse attention implementation.
    """

    def __init__(self, action_config: ActionConfig):
        super().__init__()
        self.action_config = action_config

        # Preprocessor
        self.preprocessor = MousePreprocessor(
            action_config.vae_time_compression_ratio, action_config.windows_size
        )

        # MLP to fuse hidden states with mouse condition
        mouse_input_dim = (
            action_config.img_hidden_size
            + action_config.mouse_dim_in
            * action_config.vae_time_compression_ratio
            * action_config.windows_size
        )
        self.mouse_mlp = nn.Sequential(
            nn.Linear(mouse_input_dim, action_config.mouse_hidden_dim, bias=True),
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
        self.q_norm = WanRMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()
        self.k_norm = WanRMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()

        # Output projection
        self.proj_mouse = nn.Linear(
            action_config.mouse_hidden_dim, action_config.img_hidden_size, bias=action_config.qkv_bias
        )

        # KV Cache manager
        self.kv_cache_manager = KVCacheManager(
            local_attn_size=action_config.local_attn_size, sink_size=0
        )

        # Attention core
        self.attn_core = FlashInferAttentionCore()

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        is_causal: bool = False,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        start_frame: int = 0,
        num_frame_per_block: int = 1,
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for mouse condition injection.

        Args:
            x: [B, T*S, C_img] - Input hidden states
            condition: [B, N_frames, C_mouse] - Mouse condition
            freqs_cis: (cos, sin) RoPE frequencies
            spatial_shape: (H, W) spatial dimensions
            temporal_shape: T temporal dimension
            is_causal: Whether to use causal attention
            kv_cache: KV cache for incremental decoding
            start_frame: Starting frame index for RoPE
            num_frame_per_block: Number of frames per block
            block_mask: Optional block mask

        Returns:
            Output hidden states [B, T*S, C_img]
        """
        if condition is None:
            return x

        B, T_S, C_img = x.shape
        T = temporal_shape
        H, W = spatial_shape
        S = H * W

        # Original preprocessor (commented out in favor of triton kernel)
        # hidden_states = rearrange(x, "B (T S) C -> (B S) T C", T=T, S=S)
        # fused_features = self.preprocessor(hidden_states, condition, is_causal, num_frame_per_block)

        # Fuse with mouse condition using triton kernel (accepts B, T*S, C directly)
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
        fused_features = self.mouse_mlp(fused_features)

        # QKV projection and split
        qkv = self.t_qkv(fused_features)  # [BS, T, 3*C_mouse]
        q, k, v = rearrange(
            qkv, "BS T (K H D) -> K BS T H D", K=3, H=self.action_config.heads_num
        )

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

            # Update KV cache and get window
            k_window, v_window, _, _ = self.kv_cache_manager.update_cache(kv_cache, k_rope, v, num_frame_per_block)
            # Compute attention with cached KV (RoPE already applied)
            attn_output = self.attn_core(q_rope, k_window, v_window, causal=False, use_rope=False)
        else:
            # Regular attention: use FlashInfer's integrated RoPE for best performance
            attn_output = self.attn_core(
                q, k, v,
                causal=is_causal,
                use_rope=True,
                rope_offset=start_frame
            )

        # Reshape and project: [BS, T, H, D] -> [B, T*S, C_img]
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_mouse(attn_output)

        # Residual connection
        output = x + attn_output
        return output


class KeyboardInjector(IAttentionInjector):
    """
    Keyboard condition injector using cross-attention with RoPE.
    Based on the original ActionModule keyboard attention implementation.
    """

    def __init__(self, action_config: ActionConfig):
        super().__init__()
        self.action_config = action_config

        # Preprocessor
        self.preprocessor = KeyboardPreprocessor(
            action_config.vae_time_compression_ratio,
            action_config.windows_size,
            action_config.keyboard_dim_in,
            action_config.hidden_size,
        )

        # Query projection (from hidden states)
        self.mouse_attn_q = nn.Linear(
            action_config.img_hidden_size,
            action_config.keyboard_hidden_dim,
            bias=action_config.qkv_bias,
        )

        # Key-Value projection (from keyboard condition)
        self.keyboard_attn_kv = nn.Linear(
            action_config.hidden_size
            * action_config.windows_size
            * action_config.vae_time_compression_ratio,
            action_config.keyboard_hidden_dim * 2,
            bias=action_config.qkv_bias,
        )

        # QK normalization
        head_dim = action_config.keyboard_head_dim
        self.q_norm = WanRMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()
        self.k_norm = WanRMSNorm(head_dim, eps=1e-6) if action_config.qk_norm else nn.Identity()

        # Output projection
        self.proj_keyboard = nn.Linear(
            action_config.keyboard_hidden_dim,
            action_config.img_hidden_size,
            bias=action_config.qkv_bias,
        )

        # KV Cache manager
        self.kv_cache_manager = KVCacheManager(
            local_attn_size=action_config.local_attn_size, sink_size=0
        )

        # Attention core
        self.attn_core = FlashInferAttentionCore()

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        is_causal: bool = False,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        start_frame: int = 0,
        num_frame_per_block: int = 1,
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for keyboard condition injection.

        Args:
            x: [B, T*S, C_img] - Input hidden states
            condition: [B, N_frames, C_keyboard] - Keyboard condition
            freqs_cis: (cos, sin) RoPE frequencies
            spatial_shape: (H, W) spatial dimensions
            temporal_shape: T temporal dimension
            is_causal: Whether to use causal attention
            kv_cache: KV cache for incremental decoding
            start_frame: Starting frame index for RoPE
            num_frame_per_block: Number of frames per block
            block_mask: Optional block mask

        Returns:
            Output hidden states [B, T*S, C_img]
        """
        if condition is None:
            return x

        B, T_S, C_img = x.shape
        T = temporal_shape
        H, W = spatial_shape
        S = H * W

        # Process keyboard condition
        group_keyboard = self.preprocessor(condition, is_causal, num_frame_per_block)

        # Compute Query from hidden states
        q = self.mouse_attn_q(x)  # [B, T*S, C_keyboard]
        q = q.view(B, T_S, self.action_config.heads_num, self.action_config.keyboard_head_dim)

        # Compute Key-Value from keyboard condition
        keyboard_kv = self.keyboard_attn_kv(group_keyboard)  # [B, T_k, 2*C_keyboard]
        k, v = rearrange(
            keyboard_kv,
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

        if is_causal and kv_cache is not None:
            # For Triton kernel, we need K, V in shape [S, num_new_tokens, num_heads, head_dim]
            # Current: k_rope is [B, T_k, H, D], v is [B, T_k, H, D]
            # We need to expand to spatial dimension and transpose to get the right shape

            # Expand to spatial dimension: [B, T_k, H, D] -> [B, S, T_k, H, D]
            k_rope_expanded = k_rope.unsqueeze(1).expand(-1, S, -1, -1, -1)
            v_expanded = v.unsqueeze(1).expand(-1, S, -1, -1, -1)

            # Transpose to [S, T_k, H, D] format for Triton kernel (take first batch)
            k_rope_for_cache = k_rope_expanded[0]  # [S, T_k, H, D]
            v_for_cache = v_expanded[0]  # [S, T_k, H, D]

            # Use Triton fused kernel for KV cache update with mean operation
            k_window, v_window, _, _ = update_kv_cache_triton(
                kv_cache,
                k_rope_for_cache,  # [S, T_k, H, D]
                v_for_cache,       # [S, T_k, H, D]
                max_attention_size=self.kv_cache_manager.max_attention_size,
                sink_tokens=self.kv_cache_manager.sink_tokens,
            )

            # Expand window to all spatial locations: [1, window_len, H, D] -> [B*S, window_len, H, D]
            k_window = k_window.expand(B * S, -1, -1, -1)
            v_window = v_window.expand(B * S, -1, -1, -1)

            # Compute attention with cached KV (RoPE already applied)
            attn_output = self.attn_core(q_rope, k_window, v_window, causal=False, use_rope=False)
        else:
            # Regular cross-attention
            # Expand K, V to match spatial dimension: [B, T_k, H, D] -> [B*S, T_k, H, D]
            k_rope_expanded = k_rope.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, -1, k_rope.shape[-2], k_rope.shape[-1])
            v_expanded = v.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, -1, v.shape[-2], v.shape[-1])

            # Compute cross-attention (RoPE already applied)
            attn_output = self.attn_core(q_rope, k_rope_expanded, v_expanded, causal=False, use_rope=False)

        # Reshape and project: [B*S, T, H, D] -> [B, T*S, C_img]
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_keyboard(attn_output)

        # Residual connection
        output = x + attn_output
        return output
