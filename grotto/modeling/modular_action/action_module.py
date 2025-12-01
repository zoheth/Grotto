"""
Modular Action Module - Clean implementation from first principles
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from grotto.modeling.modular_action.action_config import ActionConfig
from grotto.modeling.modular_action.movement_control import MovementInjector
from grotto.modeling.modular_action.view_control import ViewControlInjector

if TYPE_CHECKING:
    from ..ring_buffer_cache import RingBufferActionCache


@dataclass
class ActionContext:
    # Input conditions (camera control)
    rotation_cond: Optional[torch.Tensor] = None  # [B, N_frames, C_rotation] - Camera rotation
    translation_cond: Optional[
        torch.Tensor
    ] = None  # [B, N_frames, C_translation] - Camera movement

    # Per-block KV cache
    kv_cache_rotation: Optional["RingBufferActionCache"] = None
    kv_cache_translation: Optional["RingBufferActionCache"] = None

    # Runtime configuration
    num_frame_per_block: int = 1

    @property
    def has_any_condition(self) -> bool:
        """Check if any action conditioning is provided."""
        return self.rotation_cond is not None or self.translation_cond is not None


class ActionModule(nn.Module):
    """
    Action condition injector using view control and movement attention.

    Injects action conditions (view control and movement) into hidden states
    via attention mechanisms with RoPE positional encoding.

    Terminology:
        - View control: Camera/view angle control (mouse, gamepad right stick)
        - Movement: Character movement (keyboard, gamepad left stick)
    """

    def __init__(self, action_config: ActionConfig):
        """
        Initialize ActionModule.

        Args:
            action_config: Complete configuration for the action module
        """
        super().__init__()
        self.config = action_config

        # Initialize injectors based on config
        # Note: Config still uses "enable_mouse/keyboard" names for backward compatibility
        self.view_control_injector = (
            ViewControlInjector(action_config) if action_config.enable_mouse else None
        )
        self.movement_injector = (
            MovementInjector(action_config) if action_config.enable_keyboard else None
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: tuple,
        rotation: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        kv_cache_rotation: Optional["RingBufferActionCache"] = None,
        kv_cache_translation: Optional["RingBufferActionCache"] = None,
        start_frame: int = 0,
        num_frame_per_block: int = 3,
    ) -> torch.Tensor:
        """
        Inject camera control conditions into hidden states.

        Args:
            x: [B, T*H*W, C] - Hidden states
            grid_sizes: (F, H, W) - Latent grid dimensions
            rotation: [B, N_frames, C_rotation] - Camera rotation (view direction)
            translation: [B, N_frames, C_translation] - Camera translation (movement)
            is_causal: Whether to use causal attention
            kv_cache_rotation: Rotation KV cache
            kv_cache_translation: Translation KV cache
            start_frame: Starting frame index for RoPE
            num_frame_per_block: Number of frames per block

        Returns:
            [B, T*H*W, C] - Processed hidden states
        """
        tt, th, tw = grid_sizes
        x.shape[0]
        assert (
            tt * th * tw == x.shape[1]
        ), f"Sequence length mismatch: {tt}*{th}*{tw}={tt * th * tw} != {x.shape[1]}"

        hidden_states = x

        # View control injection (camera rotation)
        if self.view_control_injector is not None and rotation is not None:
            hidden_states = self.view_control_injector(
                hidden_states,
                condition=rotation,
                spatial_shape=(th, tw),
                temporal_shape=tt,
                is_causal=is_causal,
                kv_cache=kv_cache_rotation,
                start_frame=start_frame,
                num_frame_per_block=num_frame_per_block,
            )

        # Movement injection (camera translation)
        if self.movement_injector is not None and translation is not None:
            hidden_states = self.movement_injector(
                hidden_states,
                condition=translation,
                spatial_shape=(th, tw),
                temporal_shape=tt,
                is_causal=is_causal,
                kv_cache=kv_cache_translation,
                start_frame=start_frame,
                num_frame_per_block=num_frame_per_block,
            )

        return hidden_states

    def init_weights(self):
        """Initialize output projection weights to zero for residual path."""
        if self.view_control_injector is not None:
            nn.init.zeros_(self.view_control_injector.proj_view.weight)
            if self.view_control_injector.proj_view.bias is not None:
                nn.init.zeros_(self.view_control_injector.proj_view.bias)

        if self.movement_injector is not None:
            nn.init.zeros_(self.movement_injector.proj_movement.weight)
            if self.movement_injector.proj_movement.bias is not None:
                nn.init.zeros_(self.movement_injector.proj_movement.bias)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Remap legacy checkpoint keys to new modular structure."""
        # Key mappings: old_key -> new_key
        key_mappings = {
            # View control injector (legacy: mouse_injector)
            "mouse_mlp.0.weight": "view_control_injector.view_mlp.0.weight",
            "mouse_mlp.0.bias": "view_control_injector.view_mlp.0.bias",
            "mouse_mlp.2.weight": "view_control_injector.view_mlp.2.weight",
            "mouse_mlp.2.bias": "view_control_injector.view_mlp.2.bias",
            "mouse_mlp.3.weight": "view_control_injector.view_mlp.3.weight",
            "mouse_mlp.3.bias": "view_control_injector.view_mlp.3.bias",
            "t_qkv.weight": "view_control_injector.t_qkv.weight",
            "t_qkv.bias": "view_control_injector.t_qkv.bias",
            "img_attn_q_norm.weight": "view_control_injector.q_norm.weight",
            "img_attn_k_norm.weight": "view_control_injector.k_norm.weight",
            "proj_mouse.weight": "view_control_injector.proj_view.weight",
            "proj_mouse.bias": "view_control_injector.proj_view.bias",
            # Movement injector (legacy: keyboard_injector)
            "keyboard_embed.0.weight": "movement_injector.preprocessor.movement_embed.0.weight",
            "keyboard_embed.0.bias": "movement_injector.preprocessor.movement_embed.0.bias",
            "keyboard_embed.2.weight": "movement_injector.preprocessor.movement_embed.2.weight",
            "keyboard_embed.2.bias": "movement_injector.preprocessor.movement_embed.2.bias",
            "mouse_attn_q.weight": "movement_injector.q_proj.weight",
            "mouse_attn_q.bias": "movement_injector.q_proj.bias",
            "keyboard_attn_kv.weight": "movement_injector.kv_proj.weight",
            "keyboard_attn_kv.bias": "movement_injector.kv_proj.bias",
            "key_attn_q_norm.weight": "movement_injector.q_norm.weight",
            "key_attn_k_norm.weight": "movement_injector.k_norm.weight",
            "proj_keyboard.weight": "movement_injector.proj_movement.weight",
            "proj_keyboard.bias": "movement_injector.proj_movement.bias",
        }

        # Remap keys if legacy format detected
        remapped = 0
        for old_key, new_key in key_mappings.items():
            full_old_key = prefix + old_key
            if full_old_key in state_dict:
                state_dict[prefix + new_key] = state_dict.pop(full_old_key)
                remapped += 1

        if remapped > 0:
            print(f"[ActionModule] Remapped {remapped} keys from legacy checkpoint format")

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self) -> str:
        """Readable representation."""
        return (
            f"ActionModule(\n"
            f"  view_control={self.config.enable_mouse}, "
            f"movement={self.config.enable_keyboard}, "
            f"hidden_size={self.config.img_hidden_size}, "
            f"heads={self.config.heads_num}\n"
            f")"
        )
