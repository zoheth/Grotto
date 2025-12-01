"""
Conditional data processing utilities.

This module handles the preparation and slicing of conditional inputs
(visual context, action conditions) for the diffusion model.
"""

from typing import Dict, Optional

import torch

from grotto.pipeline.action_strategies import ActionDict
from grotto.pipeline.config import VAEConfig


class ConditionProcessor:
    """
    Processes and manages conditional inputs for the diffusion model.

    This includes:
    - Visual context (CLIP features)
    - Concatenated conditioning (mask + initial frame latents)
    - Action conditioning (mouse + keyboard)
    """

    def __init__(self, vae_config: VAEConfig, mode: str):
        self.vae_config = vae_config
        self.mode = mode

    def get_action_sequence_length(self, current_block_end: int) -> int:
        """
        Calculate the action conditioning sequence length up to a given block.

        The action condition has higher temporal resolution than latents.

        Args:
            current_block_end: Index of the last latent frame in the current block

        Returns:
            Action sequence length
        """
        return self.vae_config.get_action_condition_length(current_block_end)

    def slice_block_conditions(
        self,
        conditional_dict: Dict[str, torch.Tensor],
        current_start_frame: int,
        num_frames: int,
        replace_action: Optional[ActionDict] = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Extract conditions for the current block and optionally update actions.
        """
        new_cond = {}

        new_cond["cond_concat"] = conditional_dict["cond_concat"][
            :, :, current_start_frame : current_start_frame + num_frames
        ]

        # Visual context is shared across all blocks
        new_cond["visual_context"] = conditional_dict["visual_context"]

        if replace_action is not None:
            conditional_dict = self._update_action(
                conditional_dict, current_start_frame, num_frames, replace_action
            )

        current_block_end = current_start_frame + num_frames
        action_seq_len = self.get_action_sequence_length(current_block_end)

        if self.mode != "templerun":
            new_cond["rotation_cond"] = conditional_dict["rotation_cond"][:, :action_seq_len]
        new_cond["translation_cond"] = conditional_dict["translation_cond"][:, :action_seq_len]
        return new_cond, conditional_dict

    def _update_action(
        self,
        conditional_dict: Dict[str, torch.Tensor],
        current_start_frame: int,
        num_frames: int,
        replace_action: ActionDict,
    ) -> Dict[str, torch.Tensor]:
        if current_start_frame == 0:
            action_frame_count = self.vae_config.get_action_condition_length(num_frames)
        else:
            action_frame_count = self.vae_config.temporal_compression * num_frames

        final_frame = self.get_action_sequence_length(current_start_frame + num_frames)

        start_pos = final_frame - action_frame_count

        if self.mode != "templerun" and "rotation" in replace_action:
            rotation_action = replace_action["rotation"][None, None, :]  # [1, 1, action_dim]
            conditional_dict["rotation_cond"][:, start_pos:final_frame] = rotation_action.repeat(
                1, action_frame_count, 1
            )

        if "translation" in replace_action:
            translation_action = replace_action["translation"][None, None, :]  # [1, 1, action_dim]
            conditional_dict["translation_cond"][
                :, start_pos:final_frame
            ] = translation_action.repeat(1, action_frame_count, 1)

        return conditional_dict
