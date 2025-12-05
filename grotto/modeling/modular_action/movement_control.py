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
from grotto.modeling.rope import RoPE3DCache

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

        # Attention configuration
        self.num_heads = action_config.heads_num
        self.head_dim = action_config.keyboard_head_dim

        # RoPE cache
        self.rope_cache: Optional[RoPE3DCache] = None
        self.rope_cache_1d_temporal: Optional[torch.Tensor] = None  # For K's temporal-only RoPE

        # FlashInfer setup
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

    def _init_rope_cache(self, freqs: torch.Tensor):
        if self.rope_cache is None:
            self.rope_cache = RoPE3DCache(freqs=freqs, height=self.height, width=self.width)

        if self.rope_cache_1d_temporal is None:
            # Extract only temporal component for K's 1D RoPE
            device = freqs.device
            max_frames = 150

            if freqs.is_complex():
                freqs = freqs.to(device=device)
            else:
                freqs = freqs.to(dtype=torch.float32, device=device)

            head_dim_half = freqs.shape[1]
            c_height = head_dim_half // 3
            c_width = head_dim_half // 3
            c_time = head_dim_half - c_height - c_width

            # Only extract temporal component
            freqs_time = freqs[:max_frames, :c_time]

            # Pad to full head_dim_half (rest dimensions get identity rotation)
            # This ensures compatibility with apply_rope_with_cos_sin_cache
            freqs_1d = torch.zeros(max_frames, head_dim_half, dtype=freqs_time.dtype, device=device)
            freqs_1d[:, :c_time] = freqs_time

            # Compute cos and sin
            if freqs_1d.is_complex():
                cos = freqs_1d.real.float()
                sin = freqs_1d.imag.float()
            else:
                cos = torch.cos(freqs_1d)
                sin = torch.sin(freqs_1d)

            # FlashInfer format: [max_pos, rotary_dim] where first half is cos, second half is sin
            self.rope_cache_1d_temporal = torch.cat([cos, sin], dim=-1).contiguous()

    def plan_kv_and_attention(
        self,
        incoming_len: int,
        kv_cache: "DualPlaneKVCache",
        current_start: int,
        current_end: int,
        grid_sizes: Tuple[int, int, int],
        cache_mode: str = "read_write",
    ) -> None:
        """
        Delegates planning to the backend.
        """
        _, height, width = grid_sizes

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

        # Process movement condition
        group_movement = self.preprocessor(
            condition, is_causal=True, num_frame_per_block=num_frame_per_block
        )

        # Compute Query from hidden states
        q = self.q_proj(x)
        q = q.view(B, T, S, self.num_heads, self.head_dim)
        q = q.transpose(1, 2).reshape(B * S, T, self.num_heads, self.head_dim)

        # Compute Key-Value from movement condition
        movement_kv = self.kv_proj(group_movement)
        k, v = rearrange(
            movement_kv,
            "B L (K H D) -> K B L H D",
            K=2,
            H=self.num_heads,
            D=self.head_dim,
        )

        # Apply QK norm
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE
        self._init_rope_cache(freqs)
        assert self.rope_cache is not None
        assert self.rope_cache_1d_temporal is not None

        BS_q, seq_len_q, H_q, D_q = q.shape
        positions_q = torch.arange(seq_len_q, device=q.device, dtype=torch.int32) + start_frame
        positions_q = positions_q.unsqueeze(0).expand(BS_q, -1).reshape(-1)

        BS_k, seq_len_k, H_k, D_k = k.shape
        positions_k = torch.arange(seq_len_k, device=k.device, dtype=torch.int32) + start_frame
        positions_k = positions_k.unsqueeze(0).expand(BS_k, -1).reshape(-1)

        # Apply 3D RoPE to Q (queries have spatial+temporal encoding)
        cache_3d = self.rope_cache.get_cache()
        q_flat = q.reshape(BS_q * seq_len_q, H_q * D_q)
        roped_q_flat, _ = flashinfer.apply_rope_with_cos_sin_cache(
            positions_q, q_flat, q_flat, head_size=D_q, cos_sin_cache=cache_3d, is_neox=False
        )

        # Apply 1D temporal RoPE to K (keys only have temporal encoding)
        k_flat = k.reshape(BS_k * seq_len_k, H_k * D_k)
        roped_k_flat, _ = flashinfer.apply_rope_with_cos_sin_cache(
            positions_k,
            k_flat,
            k_flat,
            head_size=D_k,
            cos_sin_cache=self.rope_cache_1d_temporal,
            is_neox=False,
        )

        # Reshape for attention computation
        roped_q = roped_q_flat.view(BS_q * seq_len_q, H_q, D_q)
        roped_k = roped_k_flat.view(BS_k * seq_len_k, H_k, D_k)
        v = v.view(BS_k * seq_len_k, H_k, D_k)

        attn_output = self.attn_backend(
            roped_q=roped_q, roped_k=roped_k, v=v, kv_cache=kv_cache, cache_mode=cache_mode
        )

        attn_output = attn_output.view(B * S, T, H_q, D_q)

        # Reshape and project: [B*S, T, H, D] -> [B, T*S, C_img]
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_movement(attn_output)

        # Residual connection
        output = x + attn_output
        return output
