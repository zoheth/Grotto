"""Pose tracking and action adjustment for temporal consistency."""

from dataclasses import dataclass
from typing import List, Optional

import torch

from grotto.camera_pose import (
    CameraPose,
    PoseTransform,
    apply_transforms,
    compute_transform,
    find_closest_pose,
)


@dataclass
class PoseHistoryEntry:
    block_idx: int
    pose: CameraPose


class PoseTracker:
    """Tracks camera poses throughout inference iterations."""

    def __init__(self, initial_pose: Optional[CameraPose] = None):
        self.history: List[PoseHistoryEntry] = []
        self.current_pose = initial_pose if initial_pose else CameraPose.identity()
        self.history.append(PoseHistoryEntry(block_idx=0, pose=self.current_pose.clone()))

    def apply_block_actions(
        self,
        translations: torch.Tensor,
        rotations: torch.Tensor,
        block_idx: int,
    ) -> CameraPose:
        """Apply all actions from a block and record the final pose."""
        transforms = []
        seq_len = translations.shape[0]

        for i in range(seq_len):
            translation_3d = self._action_to_translation_3d(translations[i])
            pitch, yaw = rotations[i][0].item(), rotations[i][1].item()
            rotation_quat = self._euler_delta_to_quaternion(pitch, yaw, device=translations.device)
            transforms.append(PoseTransform(translation=translation_3d, rotation=rotation_quat))

        self.current_pose = apply_transforms(self.current_pose, transforms)
        self.history.append(PoseHistoryEntry(block_idx=block_idx, pose=self.current_pose.clone()))

        return self.current_pose

    def find_closest_historical_pose(
        self, target_pose: CameraPose, search_start: int = 0, search_end: Optional[int] = None
    ) -> tuple[int, PoseHistoryEntry, float]:
        """Find the closest historical pose to the target."""
        if len(self.history) <= 1:
            return 0, self.history[0], 0.0

        if search_end is None:
            search_end = len(self.history) - 1
        else:
            search_end = min(search_end, len(self.history) - 1)

        if search_start >= search_end:
            return -1, self.history[0], float("inf")

        historical_poses = [entry.pose for entry in self.history[search_start:search_end]]

        idx, closest_pose, distance = find_closest_pose(
            target_pose,
            historical_poses,
            position_weight=1.0,
            rotation_weight=1.0,
        )

        actual_idx = search_start + idx
        return actual_idx, self.history[actual_idx], distance

    def truncate_after_block(self, block_idx: int) -> None:
        """Remove all history entries after the specified block."""
        self.history = [entry for entry in self.history if entry.block_idx <= block_idx]
        if self.history:
            self.current_pose = self.history[-1].pose.clone()

    def _action_to_translation_3d(self, action: torch.Tensor) -> torch.Tensor:
        """Convert action space translation to 3D translation."""
        if action.shape[0] == 4:
            forward = action[0].item() - action[1].item()
            left = action[2].item() - action[3].item()
            return torch.tensor([forward, left, 0.0], dtype=torch.float32, device=action.device)
        else:
            return action[:3]

    def _euler_delta_to_quaternion(
        self, pitch: float, yaw: float, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Convert pitch/yaw deltas to quaternion (roll=0)."""
        from grotto.camera_pose import euler_to_quaternion

        return euler_to_quaternion(torch.tensor([0.0, pitch, yaw], device=device))


class PoseActionAdjuster:
    """Adjusts actions based on pose history to maintain temporal consistency."""

    def __init__(self, pose_tracker: PoseTracker, max_history_length: int = 8):
        self.pose_tracker = pose_tracker
        self.max_history_length = max_history_length

    def adjust_first_action_and_get_cache_truncation(
        self,
        first_action_translation: torch.Tensor,
        first_action_rotation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """Adjust the first action to align with closest historical pose."""
        translation_3d = self.pose_tracker._action_to_translation_3d(first_action_translation)
        pitch, yaw = first_action_rotation[0].item(), first_action_rotation[1].item()
        rotation_quat = self.pose_tracker._euler_delta_to_quaternion(
            pitch, yaw, device=first_action_translation.device
        )

        transform = PoseTransform(translation=translation_3d, rotation=rotation_quat)
        predicted_pose = apply_transforms(self.pose_tracker.current_pose, [transform])

        current_history_length = len(self.pose_tracker.history)
        search_start = 2
        search_end = None

        if current_history_length > self.max_history_length:
            search_start = 2
            search_end = self.max_history_length - 1
            print(
                f"[PoseAdjuster] History length {current_history_length} > {self.max_history_length}, "
                f"limiting search to blocks {search_start}-{search_end-1}"
            )

        hist_idx, hist_entry, distance = self.pose_tracker.find_closest_historical_pose(
            predicted_pose, search_start=search_start, search_end=search_end
        )
        if hist_idx == -1:
            return first_action_translation, first_action_rotation, None

        adjusted_transform = compute_transform(hist_entry.pose, predicted_pose)
        adjusted_translation = self._translation_3d_to_action(adjusted_transform.translation)
        adjusted_rotation = self._quaternion_to_euler_delta(adjusted_transform.rotation)

        truncate_block_idx = hist_entry.block_idx

        return adjusted_translation, adjusted_rotation, truncate_block_idx

    def _translation_3d_to_action(self, translation_3d: torch.Tensor) -> torch.Tensor:
        """Convert 3D translation back to action space."""
        x, y = translation_3d[0].item(), translation_3d[1].item()

        forward = max(x, 0.0)
        back = max(-x, 0.0)
        left = max(y, 0.0)
        right = max(-y, 0.0)

        return torch.tensor(
            [forward, back, left, right], dtype=torch.float32, device=translation_3d.device
        )

    def _quaternion_to_euler_delta(self, quaternion: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to pitch/yaw deltas."""
        from grotto.camera_pose import quaternion_to_euler

        euler = quaternion_to_euler(quaternion)
        return torch.tensor(
            [euler[1].item(), euler[2].item()], dtype=torch.float32, device=quaternion.device
        )
