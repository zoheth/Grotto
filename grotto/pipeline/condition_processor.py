"""
Conditional data processing utilities.

This module handles the preparation and slicing of conditional inputs
(visual context, action conditions) for the diffusion model.
"""

from grotto.pipeline.config import VAEConfig
from grotto.types import ConditionalInputs


class ConditionProcessor:
    """
    Processes and manages conditional inputs for the diffusion model.

    This includes:
    - Visual context (CLIP features)
    - Concatenated conditioning (mask + initial frame latents)
    - Action conditioning (camera rotation and translation)
    """

    def __init__(self, vae_config: VAEConfig):
        self.vae_config = vae_config

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
        conditional_inputs: ConditionalInputs,
        current_start_frame: int,
        num_frames: int,
    ) -> tuple[ConditionalInputs, ConditionalInputs]:
        current_block_end = current_start_frame + num_frames
        action_seq_len = self.get_action_sequence_length(current_block_end)

        block_cond = ConditionalInputs(
            cond_concat=conditional_inputs.cond_concat[:, :, current_start_frame:current_block_end],
            visual_context=conditional_inputs.visual_context,
            rotation_cond=conditional_inputs.rotation_cond[:, :action_seq_len]
            if conditional_inputs.rotation_cond is not None
            else None,
            translation_cond=conditional_inputs.translation_cond[:, :action_seq_len]
            if conditional_inputs.translation_cond is not None
            else None,
        )

        return block_cond, conditional_inputs
