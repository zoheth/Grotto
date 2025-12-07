"""
Movement Control Module - Handles character movement conditioning.

This module implements movement control injection (originally keyboard-based input).
It uses cross-attention with RoPE to incorporate movement commands into the model.

Terminology:
    Movement Control = Character movement = Position change
    Input sources: Keyboard (WASD), gamepad left stick, etc.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import flashinfer
import torch
from einops import rearrange
from torch import nn

from grotto.modeling.attention import AttentionWithCache
from grotto.modeling.modular_action.action_config import ActionConfig
from grotto.modeling.modular_action.interfaces import ActionInjector

if TYPE_CHECKING:
    from ..kv_cache import DualPlaneKVCache


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

        self.num_heads = action_config.heads_num
        self.head_dim = action_config.keyboard_head_dim

        if workspace_buffer is None:
            workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        self.kv_indptr = None
        self.qo_indptr = None
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
        kv_cache: "DualPlaneKVCache",
        current_start: int,
        current_end: int,
        grid_sizes: Tuple[int, int, int],  # Unused but kept for interface consistency
        cache_mode: str = "read_write",
    ) -> None:
        self.attn_backend.plan(
            incoming_len=incoming_len,
            kv_cache=kv_cache,
            current_start=current_start,
            current_end=current_end,
            frame_seqlen=1,
            cache_mode=cache_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        freqs: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        kv_cache: Optional["DualPlaneKVCache"] = None,
        start_frame: int = 0,
        num_frame_per_block: int = 3,
        cache_mode: str = "read_write",
    ) -> torch.Tensor:
        """
        Forward pass for movement control condition injection.

        Args:
            x: [B, T*S, C_img] - Input hidden states
            condition: [B, N_frames, C_movement] - Movement control condition
            spatial_shape: (H, W) spatial dimensions
            temporal_shape: T temporal dimension
            freqs_cos: RoPE cos frequencies
            freqs_sin: RoPE sin frequencies
            kv_cache: DualPlaneKVCache for incremental decoding
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

        movement_features = self.preprocessor(
            condition, is_causal=True, num_frame_per_block=num_frame_per_block
        )

        # Query from visual features: [B, T*S, C] -> [B*S, T, H, D]
        q = self.q_proj(x)
        q = q.view(B, T, S, self.num_heads, self.head_dim)
        q = q.transpose(1, 2).reshape(B * S, T, self.num_heads, self.head_dim)

        # Key-Value from movement condition: [B, L, C] -> [B, L, H, D]
        movement_kv = self.kv_proj(movement_features)
        k, v = rearrange(movement_kv, "B L (K H D) -> K B L H D", K=2, H=self.num_heads)

        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply 1D temporal RoPE to Q and K separately (cross-attention)
        rope_theta = getattr(self.action_config, "rope_theta", 256.0)

        # Q: [B*S, T_q, H, D] - each spatial position is independent sequence
        BS, T_q, H_q, D_q = q.shape
        q_ragged = q.reshape(BS * T_q, H_q, D_q)
        indptr_q = torch.arange(0, BS + 1, dtype=torch.int32, device=q.device) * T_q
        offsets_q = torch.full((BS,), start_frame, dtype=torch.int32, device=q.device)
        roped_q, _ = flashinfer.apply_rope(
            q_ragged, q_ragged, indptr_q, offsets_q, interleave=False, rope_theta=rope_theta
        )

        # K: [B, T_k, H, D] - movement sequences
        B, T_k, H_k, D_k = k.shape
        k_ragged = k.reshape(B * T_k, H_k, D_k)
        indptr_k = torch.arange(0, B + 1, dtype=torch.int32, device=k.device) * T_k
        offsets_k = torch.full((B,), start_frame, dtype=torch.int32, device=k.device)
        _, roped_k = flashinfer.apply_rope(
            k_ragged, k_ragged, indptr_k, offsets_k, interleave=False, rope_theta=rope_theta
        )

        v = v.reshape(B * T_k, H_k, D_k)

        attn_output = self.attn_backend(
            roped_q=roped_q, roped_k=roped_k, v=v, kv_cache=kv_cache, cache_mode=cache_mode
        )

        # [BS*T, H, D] -> [BS, T, H, D] -> [B, T*S, H*D]
        attn_output = attn_output.view(B * S, T, H_q, D_q)
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_movement(attn_output)

        return x + attn_output
