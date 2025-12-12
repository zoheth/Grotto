"""
Pose-aware pipeline with temporal consistency checks.

Extends the batch pipeline to track poses and adjust actions for consistency.
"""

from typing import TYPE_CHECKING, List, Optional

import torch
from tqdm import tqdm

from grotto.camera_pose import CameraPose
from grotto.pipeline.batch_pipeline import BatchCausalInferencePipeline
from grotto.pipeline.config import PipelineConfig
from grotto.pipeline.pose_tracker import PoseActionAdjuster, PoseTracker
from grotto.types import ConditionalInputs

if TYPE_CHECKING:
    from grotto.modeling.predictor import WanDiffusionPredictor
    from grotto.modeling.vae_wrapper import VaeDecoderWrapper


class PoseAwarePipeline(BatchCausalInferencePipeline):
    """
    Pipeline with pose tracking and action adjustment for temporal consistency.

    This pipeline:
    1. Tracks camera pose throughout inference
    2. Before each block, checks if first action leads to pose close to history
    3. If close match found, adjusts action to match exactly and truncates cache
    """

    def __init__(
        self,
        config: PipelineConfig,
        predictor: "WanDiffusionPredictor",
        vae_decoder: "VaeDecoderWrapper",
        device: str = "cuda",
        initial_pose: Optional[CameraPose] = None,
        enable_pose_adjustment: bool = True,
        max_history_length: int = 8,
        distance_threshold: float = 0.1,
    ):
        """
        Initialize pose-aware pipeline.

        Args:
            config: Pipeline configuration
            predictor: Diffusion predictor
            vae_decoder: VAE decoder
            device: Device to run on
            initial_pose: Starting camera pose
            enable_pose_adjustment: Enable pose-based action adjustment
            max_history_length: Maximum history length to maintain (default: 8)
            distance_threshold: Distance threshold for triggering rollback (default: 0.1)
                Rollback only occurs when predicted pose distance < threshold
        """
        super().__init__(config, predictor, vae_decoder, device)
        self.pose_tracker = PoseTracker(initial_pose)
        self.pose_adjuster = PoseActionAdjuster(
            self.pose_tracker, max_history_length, distance_threshold
        )
        self.enable_pose_adjustment = enable_pose_adjustment
        self.max_history_length = max_history_length
        self.distance_threshold = distance_threshold

    def _extract_action_at_index(
        self, conditional_inputs: ConditionalInputs, action_index: int, batch_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract translation and rotation at a specific action index.

        Args:
            conditional_inputs: Full conditional inputs
            action_index: Index in action sequence
            batch_idx: Batch index to extract from

        Returns:
            Tuple of (translation, rotation)
        """
        translation = conditional_inputs.translation_cond[batch_idx, action_index]
        rotation = (
            conditional_inputs.rotation_cond[batch_idx, action_index]
            if conditional_inputs.rotation_cond is not None
            else torch.zeros(2)
        )
        return translation, rotation

    def _update_action_at_index(
        self,
        conditional_inputs: ConditionalInputs,
        action_index: int,
        new_translation: torch.Tensor,
        new_rotation: torch.Tensor,
        batch_idx: int = 0,
    ) -> ConditionalInputs:
        """
        Update action at a specific index in conditional inputs.

        Args:
            conditional_inputs: Conditional inputs to modify
            action_index: Index to update
            new_translation: New translation action
            new_rotation: New rotation action
            batch_idx: Batch index to update

        Returns:
            Updated conditional inputs (modified in-place)
        """
        conditional_inputs.translation_cond[batch_idx, action_index] = new_translation
        if conditional_inputs.rotation_cond is not None:
            conditional_inputs.rotation_cond[batch_idx, action_index] = new_rotation
        return conditional_inputs

    def _truncate_cache_at_action_index(self, action_index: int, current_logical_frame: int):
        """
        Truncate KV cache to correspond to a specific action index.

        Args:
            action_index: Action index to truncate after
            current_logical_frame: Current logical frame position
        """
        # Convert action index to frame index
        # action_length = 1 + temporal_compression * (frames - 1)
        # Solving for frames: frames = (action_length - 1) / temporal_compression + 1
        temporal_compression = self.config.vae.temporal_compression
        target_action_length = action_index + 1
        target_frames = (target_action_length - 1) // temporal_compression + 1

        # Calculate how many frames to pop
        frames_to_keep = target_frames
        current_frames = current_logical_frame
        frames_to_pop = current_frames - frames_to_keep

        if frames_to_pop > 0:
            print(frames_to_pop)
            visual_cache = self.cache_manager.get_caches()
            for cache in visual_cache:
                cache.pop_latent(frames_to_pop)

            return frames_to_keep  # New logical frame position
        return current_logical_frame

    @torch.no_grad()
    def inference(
        self,
        noise: torch.Tensor,
        conditional_inputs: ConditionalInputs,
        return_latents: bool = False,
        profile: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Run inference with pose-aware action adjustment.

        This extends the base inference to:
        1. Track poses throughout generation
        2. Adjust actions when they lead to poses close to history
        3. Truncate cache accordingly
        """
        assert noise.shape[1] == self.config.vae.latent_channels
        batch_size, num_channels, num_frames, height, width = noise.shape

        assert num_frames % self.config.inference.num_frame_per_block == 0
        num_blocks = num_frames // self.config.inference.num_frame_per_block

        self._ensure_cache_initialized(batch_size, noise.dtype)

        num_output_frames = num_frames

        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=self.device,
            dtype=noise.dtype,
        )

        from grotto.modeling.constant import ZERO_VAE_CACHE

        vae_cache = [None] * len(ZERO_VAE_CACHE)
        videos = []

        current_start_frame = 0
        logical_frame_position = 0
        current_action_index = 0

        all_num_frames = [self.config.inference.num_frame_per_block] * num_blocks

        for block_idx, current_num_frames in enumerate(tqdm(all_num_frames)):
            # Calculate action sequence for this block
            block_end_frame = logical_frame_position + current_num_frames
            action_seq_len = self.condition_processor.get_action_sequence_length(block_end_frame)
            block_start_action = current_action_index
            block_end_action = block_start_action + (
                action_seq_len
                - self.condition_processor.get_action_sequence_length(logical_frame_position)
            )

            # Pose adjustment logic (only after first few blocks)
            if self.enable_pose_adjustment and block_idx >= 2:
                # Extract first action of this block
                first_translation, first_rotation = self._extract_action_at_index(
                    conditional_inputs, block_start_action, batch_idx=0
                )

                # Check if adjustment needed
                (
                    adjusted_translation,
                    adjusted_rotation,
                    truncate_block_idx,
                ) = self.pose_adjuster.adjust_first_action_and_get_cache_truncation(
                    first_translation, first_rotation
                )

                # Apply adjustment if needed
                if truncate_block_idx is not None:
                    # Update first action
                    self._update_action_at_index(
                        conditional_inputs,
                        block_start_action,
                        adjusted_translation,
                        adjusted_rotation,
                        batch_idx=0,
                    )

                    # Truncate cache and pose history
                    # Calculate the frame position corresponding to truncate_block_idx
                    # Each block processes current_num_frames frames
                    truncate_frame = (truncate_block_idx + 1) * current_num_frames

                    # Pop cache to go back to truncate_frame
                    frames_to_pop = logical_frame_position - truncate_frame
                    if frames_to_pop > 0:
                        print(frames_to_pop)
                        visual_cache = self.cache_manager.get_caches()
                        for cache in visual_cache:
                            cache.pop_latent(frames_to_pop // 3)

                    self.pose_tracker.truncate_after_block(truncate_block_idx)
                    logical_frame_position = truncate_frame

                    print(
                        f"[Block {block_idx}] Adjusted action & truncated to block {truncate_block_idx}"
                    )

            # Standard batch pipeline logic
            noisy_input = noise[
                :, :, current_start_frame : current_start_frame + current_num_frames
            ]

            block_cond, _ = self.condition_processor.slice_block_conditions(
                conditional_inputs, current_start_frame, current_num_frames
            )

            denoised_pred = self._denoise_block(
                noisy_input, block_cond, logical_frame_position, batch_size
            )

            output[
                :, :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred

            self._update_kv_cache_with_clean_context(
                denoised_pred, block_cond, logical_frame_position, batch_size
            )

            video, vae_cache = self._decode_latent_to_video(denoised_pred, vae_cache)
            videos.append(video)

            # Update pose tracker with NEW actions from this block only
            if block_cond.translation_cond is not None:
                # Extract only the new actions for this block
                num_new_actions = block_end_action - block_start_action
                if num_new_actions > 0:
                    # block_cond contains actions from 0 to block_end_action
                    # We want only the last num_new_actions
                    new_translations = block_cond.translation_cond[0][-num_new_actions:]
                    new_rotations = (
                        block_cond.rotation_cond[0][-num_new_actions:]
                        if block_cond.rotation_cond is not None
                        else torch.zeros(num_new_actions, 2)
                    )
                    # Apply all actions and record final pose for this block
                    self.pose_tracker.apply_block_actions(
                        translations=new_translations,
                        rotations=new_rotations,
                        block_idx=block_idx,
                    )

            current_start_frame += current_num_frames
            logical_frame_position += current_num_frames
            current_action_index = block_end_action

        if return_latents:
            return output
        else:
            return videos
