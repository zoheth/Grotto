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

    Frame Terminology:
    - Video frames: Original video frames (e.g., 1920x1080 images)
    - Latent frames: VAE-encoded frames (compressed representation)
    - Relationship: 1 latent frame = 4 video frames (temporal_compression=4)

    Pipeline Structure:
    - Inference is divided into blocks
    - Each block generates num_frame_per_block latent frames (typically 3)
    - 1 block = 1 KV cache unit = 3 latent frames = 12 video frames
    - logical_frame_position tracks total latent frames generated so far
    - cache.pop_latent(n) pops n BLOCKS (not latent frames), so conversion needed
    """

    def __init__(
        self,
        config: PipelineConfig,
        predictor: "WanDiffusionPredictor",
        vae_decoder: "VaeDecoderWrapper",
        device: str = "cuda",
        initial_pose: Optional[CameraPose] = None,
        enable_pose_adjustment: bool = True,
        max_history_length: int = 12,
        distance_threshold: float = 0.05,
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
            distance_threshold: Distance threshold for triggering rollback (default: 0.05)
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
        conditional_inputs.translation_cond[batch_idx, action_index] = new_translation
        if conditional_inputs.rotation_cond is not None:
            conditional_inputs.rotation_cond[batch_idx, action_index] = new_rotation
        return conditional_inputs

    def _insert_compensation_actions(
        self,
        conditional_inputs: ConditionalInputs,
        insert_index: int,
        translations: list[torch.Tensor],
        rotations: list[torch.Tensor],
    ) -> None:
        if conditional_inputs.translation_cond is None:
            return

        batch_size = conditional_inputs.translation_cond.shape[0]
        device = conditional_inputs.translation_cond.device
        dtype = conditional_inputs.translation_cond.dtype

        new_trans = (
            torch.stack(translations, dim=0)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .to(device=device, dtype=dtype)
        )

        conditional_inputs.translation_cond = torch.cat(
            [
                conditional_inputs.translation_cond[:, :insert_index],
                new_trans,
                conditional_inputs.translation_cond[:, insert_index + 1 :],
            ],
            dim=1,
        )

        if conditional_inputs.rotation_cond is not None:
            new_rot = (
                torch.stack(rotations, dim=0)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .to(device=device, dtype=conditional_inputs.rotation_cond.dtype)
            )
            conditional_inputs.rotation_cond = torch.cat(
                [
                    conditional_inputs.rotation_cond[:, :insert_index],
                    new_rot,
                    conditional_inputs.rotation_cond[:, insert_index + 1 :],
                ],
                dim=1,
            )

    def _latent_frames_per_block(self) -> int:
        """
        Get the number of latent frames generated per block.

        Returns:
            Number of latent frames per block
        """
        return self.config.inference.num_frame_per_block

    def _block_to_latent_frames(self, block_idx: int) -> int:
        """
        Convert block index to number of latent frames.

        Args:
            block_idx: Block index (0-indexed)

        Returns:
            Number of latent frames up to and including this block
        """
        return (block_idx + 1) * self._latent_frames_per_block()

    def _pop_latent_frames_from_cache(self, num_latent_frames: int) -> None:
        """
        Pop latent frames from KV cache.

        Note: cache.pop_latent() operates in BLOCK units, where 1 block = num_frame_per_block latent frames.
        This method converts latent frames to blocks before popping.

        Args:
            num_latent_frames: Number of latent frames to remove
        """
        if num_latent_frames <= 0:
            return

        # Convert latent frames to blocks
        # 1 block = num_frame_per_block latent frames (typically 3)
        num_blocks_to_pop = num_latent_frames // self._latent_frames_per_block()

        if num_blocks_to_pop <= 0:
            return

        visual_cache = self.cache_manager.get_caches()
        for cache in visual_cache:
            cache.pop_latent(num_blocks_to_pop)
        print(
            f"[Cache] Popped {num_blocks_to_pop} blocks "
            f"({num_blocks_to_pop * self._latent_frames_per_block()} latent frames) from cache"
        )

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

        Args:
            noise: Latent noise tensor [batch, channels, latent_frames, height, width]
            conditional_inputs: Action conditions aligned with video frames
            return_latents: If True, return latent tensors; otherwise decoded videos
            profile: Enable profiling (not implemented)

        Returns:
            List of decoded video tensors or latent tensors
        """
        assert noise.shape[1] == self.config.vae.latent_channels
        batch_size, num_channels, num_frames, height, width = noise.shape

        # num_frames here refers to LATENT frames (not video frames)
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

        logical_frame_position = 0
        current_action_index = 0
        block_idx = 0
        # rollback_triggered = False

        pbar = tqdm(total=num_blocks)

        while logical_frame_position < num_frames:
            latent_frames_per_block = self.config.inference.num_frame_per_block
            next_frame_position = logical_frame_position + latent_frames_per_block

            if conditional_inputs.translation_cond is not None:
                num_new_actions = self.condition_processor.get_action_sequence_length(
                    next_frame_position
                ) - self.condition_processor.get_action_sequence_length(logical_frame_position)
                remaining_actions = (
                    conditional_inputs.translation_cond.shape[1] - current_action_index
                )

                if remaining_actions < num_new_actions:
                    print(
                        f"[Early termination] Not enough actions remaining: "
                        f"need {num_new_actions}, only {remaining_actions} left at frame {logical_frame_position}"
                    )
                    break
            else:
                num_new_actions = 0

            if (
                self.enable_pose_adjustment
                and block_idx >= 2
                and conditional_inputs.translation_cond is not None
                and current_action_index < conditional_inputs.translation_cond.shape[1]
            ):
                first_translation, first_rotation = self._extract_action_at_index(
                    conditional_inputs, current_action_index
                )

                (
                    translations,
                    rotations,
                    rollback_frame_position,
                    _,
                ) = self.pose_adjuster.adjust_first_action_and_get_cache_truncation(
                    first_translation, first_rotation
                )

                if rollback_frame_position is not None:
                    self._pop_latent_frames_from_cache(
                        logical_frame_position - rollback_frame_position
                    )
                    self.pose_tracker.truncate_to_frame_position(rollback_frame_position)
                    logical_frame_position = rollback_frame_position
                    # rollback_triggered = True

                    self._insert_compensation_actions(
                        conditional_inputs, current_action_index, translations, rotations
                    )

                    next_frame_position = logical_frame_position + latent_frames_per_block
                    num_new_actions = self.condition_processor.get_action_sequence_length(
                        next_frame_position
                    ) - self.condition_processor.get_action_sequence_length(logical_frame_position)

                    print(
                        f"[Rollback] Block {block_idx}, "
                        f"latent_frames={logical_frame_position}, action_index={current_action_index}"
                    )

            block_cond, _ = self.condition_processor.slice_block_conditions(
                conditional_inputs,
                logical_frame_position,
                latent_frames_per_block,
                current_action_index,
            )

            noisy_input = noise[
                :, :, logical_frame_position : logical_frame_position + latent_frames_per_block
            ]
            denoised_pred = self._denoise_block(
                noisy_input, block_cond, logical_frame_position, batch_size
            )
            output[
                :, :, logical_frame_position : logical_frame_position + latent_frames_per_block
            ] = denoised_pred

            self._update_kv_cache_with_clean_context(
                denoised_pred, block_cond, logical_frame_position, batch_size
            )

            video, vae_cache = self._decode_latent_to_video(denoised_pred, vae_cache)
            # if rollback_triggered:
            #     video = video[:, 8:, :, :, :]
            #     rollback_triggered = False
            videos.append(video)

            # Add black separator frames between blocks for easy visualization
            if block_idx < num_blocks - 1:  # Don't add separator after the last block
                num_separator_frames = 3  # Number of black frames to insert
                # video shape: [batch, frames, channels, height, width]
                batch_size_v, _, num_channels_v, height_v, width_v = video.shape
                black_frames = torch.zeros(
                    (batch_size_v, num_separator_frames, num_channels_v, height_v, width_v),
                    device=video.device,
                    dtype=video.dtype,
                )
                videos.append(black_frames)

            if block_cond.translation_cond is not None and num_new_actions > 0:
                new_translations = block_cond.translation_cond[0][-num_new_actions:]
                new_rotations = (
                    block_cond.rotation_cond[0][-num_new_actions:]
                    if block_cond.rotation_cond is not None
                    else torch.zeros(num_new_actions, 2, device=new_translations.device)
                )
                self.pose_tracker.apply_block_actions(
                    translations=new_translations,
                    rotations=new_rotations,
                    block_idx=block_idx,
                    logical_frame_position=logical_frame_position,
                    action_index=current_action_index,
                )

            current_action_index += num_new_actions

            logical_frame_position += latent_frames_per_block
            block_idx += 1
            pbar.update(1)

        pbar.close()

        if return_latents:
            return output
        else:
            return videos
