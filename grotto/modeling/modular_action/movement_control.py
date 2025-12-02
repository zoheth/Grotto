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

from grotto.modeling.modular_action.action_config import ActionConfig
from grotto.modeling.modular_action.interfaces import ActionInjector

if TYPE_CHECKING:
    from ..kv_cache import DualPlaneKVCache


class RoPE3DCache:
    def __init__(self, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        self.global_cache = torch.cat([freqs_cos, freqs_sin], dim=-1).contiguous()

    def get_cache(self) -> torch.Tensor:
        return self.global_cache


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
        self, action_config: ActionConfig, workspace_buffer: Optional[torch.Tensor] = None
    ):
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

        # Attention configuration
        self.num_heads = action_config.heads_num
        self.head_dim = action_config.keyboard_head_dim

        # RoPE cache
        self.rope_cache: Optional[RoPE3DCache] = None

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

    def _init_rope_cache(self, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        if self.rope_cache is None:
            self.rope_cache = RoPE3DCache(freqs_cos=freqs_cos, freqs_sin=freqs_sin)

    def plan_attention(
        self,
        num_spatial: int,
        q_len: int,
        kv_len: int,
        kv_cache: "DualPlaneKVCache",
        current_start: int,
        current_end: int,
        cache_mode: str = "read_write",
    ):
        if cache_mode == "read_write":
            kv_cache.plan_append(kv_len)
            # Keep last max_attention_size tokens (no spatial dimension for cross-attn K/V)
            keep_size = self.max_attention_size
            kv_cache.evict(keep_size)
            kv_len_total = kv_cache.total_tokens
        elif cache_mode == "read_only":
            # Don't modify cache state in read_only mode
            kv_len_total = kv_cache.total_tokens + kv_len
        else:
            raise ValueError(f"Invalid cache_mode: {cache_mode}")

        # total_q_len = num_spatial * q_len
        if self.qo_indptr is None or self.qo_indptr.shape[0] != num_spatial + 1:
            self.qo_indptr = torch.arange(
                0, (num_spatial + 1) * q_len, q_len, dtype=torch.int32, device="cuda"
            )
        else:
            torch.arange(
                0,
                (num_spatial + 1) * q_len,
                q_len,
                dtype=torch.int32,
                device="cuda",
                out=self.qo_indptr,
            )

        if self.kv_indptr is None or self.kv_indptr.shape[0] != num_spatial + 1:
            self.kv_indptr = torch.arange(
                0, (num_spatial + 1) * kv_len_total, kv_len_total, dtype=torch.int32, device="cuda"
            )
        else:
            torch.arange(
                0,
                (num_spatial + 1) * kv_len_total,
                kv_len_total,
                dtype=torch.int32,
                device="cuda",
                out=self.kv_indptr,
            )

        self.flashinfer_wrapper.plan(
            self.qo_indptr,
            self.kv_indptr,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            q_data_type=torch.bfloat16,
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        kv_cache: Optional["DualPlaneKVCache"] = None,
        start_frame: int = 0,
        num_frame_per_block: int = 1,
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
        self._init_rope_cache(freqs_cos, freqs_sin)
        assert self.rope_cache is not None

        BS_q, seq_len_q, H_q, D_q = q.shape
        positions_q = torch.arange(seq_len_q, device=q.device, dtype=torch.int32) + start_frame
        positions_q = positions_q.unsqueeze(0).expand(BS_q, -1).reshape(-1)

        BS_k, seq_len_k, H_k, D_k = k.shape
        positions_k = torch.arange(seq_len_k, device=k.device, dtype=torch.int32) + start_frame
        positions_k = positions_k.unsqueeze(0).expand(BS_k, -1).reshape(-1)

        cache = self.rope_cache.get_cache()
        q_flat = q.reshape(BS_q * seq_len_q, H_q * D_q)
        k_flat = k.reshape(BS_k * seq_len_k, H_k * D_k)

        # For cross-attention, Q and K have different shapes, so apply RoPE separately
        # FlashInfer's apply_rope_with_cos_sin_cache requires Q and K to have same shape[0]
        # Workaround: pass the same tensor for both Q and K, only use first output
        roped_q_flat, _ = flashinfer.apply_rope_with_cos_sin_cache(
            positions_q, q_flat, q_flat, head_size=D_q, cos_sin_cache=cache, is_neox=False
        )
        roped_k_flat, _ = flashinfer.apply_rope_with_cos_sin_cache(
            positions_k, k_flat, k_flat, head_size=D_k, cos_sin_cache=cache, is_neox=False
        )

        # Keep batch dimension for cross-attention K/V cache (ndim=4)
        roped_q = roped_q_flat.view(BS_q, seq_len_q, H_q, D_q)
        roped_k = roped_k_flat.view(BS_k, seq_len_k, H_k, D_k)
        v_batched = v  # Already [BS_k, seq_len_k, H_k, D_k]

        # KV cache operations
        if kv_cache is not None:
            if cache_mode == "read_only":
                read_plan = kv_cache.get_read_plan()
                k_linear, v_linear = kv_cache._storage.execute_gather_with_append(
                    read_plan, roped_k, v_batched
                )
            elif cache_mode == "read_write":
                kv_cache.execute_append(roped_k, v_batched)
                k_linear, v_linear = kv_cache.get_linear_view()
            else:
                raise ValueError(f"Invalid cache_mode: {cache_mode}")

            # k_linear, v_linear: [B, kv_len, H, D]
            # Expand across S spatial locations and flatten: [B, kv_len, H, D] -> [S*kv_len, H, D]
            k_linear_expanded = (
                k_linear.unsqueeze(0).expand(S, -1, -1, -1, -1).reshape(-1, H_k, D_k)
            )
            v_linear_expanded = (
                v_linear.unsqueeze(0).expand(S, -1, -1, -1, -1).reshape(-1, H_k, D_k)
            )

            # Flatten Q for FlashInfer: [BS_q, seq_len_q, H, D] -> [BS_q*seq_len_q, H, D]
            roped_q_flat = roped_q.reshape(-1, H_q, D_q)
            attn_output = self.flashinfer_wrapper.run(
                roped_q_flat, k_linear_expanded, v_linear_expanded
            )
        else:
            # No cache: expand K/V and flatten
            # roped_k, v_batched: [B, seq_len_k, H, D]
            roped_k_expanded = roped_k.unsqueeze(0).expand(S, -1, -1, -1, -1).reshape(-1, H_k, D_k)
            v_expanded = v_batched.unsqueeze(0).expand(S, -1, -1, -1, -1).reshape(-1, H_k, D_k)

            roped_q_flat = roped_q.reshape(-1, H_q, D_q)
            attn_output = self.flashinfer_wrapper.run(roped_q_flat, roped_k_expanded, v_expanded)

        attn_output = attn_output.view(B * S, T, H_q, D_q)

        # Reshape and project: [B*S, T, H, D] -> [B, T*S, C_img]
        attn_output = rearrange(attn_output, "(B S) T H D -> B (T S) (H D)", B=B, S=S)
        attn_output = self.proj_movement(attn_output)

        # Residual connection
        output = x + attn_output
        return output
