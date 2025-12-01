from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class TranslationMapping:
    """Defines translation (movement) action mappings."""

    action_to_index: Dict[str, int]
    dimension: int

    @classmethod
    def create_wasd(cls) -> "TranslationMapping":
        """Standard WASD movement: forward/back/left/right."""
        return cls(
            action_to_index={
                "forward": 0,
                "back": 1,
                "left": 2,
                "right": 3,
            },
            dimension=4,
        )


@dataclass
class RotationMapping:
    action_to_delta: Dict[str, Tuple[float, float]]  # (pitch, yaw)
    default_sensitivity: float = 0.1

    @classmethod
    def create_standard(cls, sensitivity: float = 0.1) -> "RotationMapping":
        return cls(
            action_to_delta={
                "look_up": (sensitivity, 0),
                "look_down": (-sensitivity, 0),
                "look_left": (0, -sensitivity),
                "look_right": (0, sensitivity),
                "look_up_right": (sensitivity, sensitivity),
                "look_up_left": (sensitivity, -sensitivity),
                "look_down_right": (-sensitivity, sensitivity),
                "look_down_left": (-sensitivity, -sensitivity),
            },
            default_sensitivity=sensitivity,
        )


@dataclass
class CameraControlConfig:
    translation: TranslationMapping
    rotation: RotationMapping | None = None

    @property
    def has_rotation(self) -> bool:
        return self.rotation is not None

    @property
    def translation_dim(self) -> int:
        return self.translation.dimension

    @property
    def rotation_dim(self) -> int:
        return 2 if self.has_rotation else 0


STANDARD_CAMERA_CONFIG = CameraControlConfig(
    translation=TranslationMapping.create_wasd(),
    rotation=RotationMapping.create_standard(),
)


class CameraControlSequence:
    def __init__(self, config: CameraControlConfig, total_frames: int):
        self.config = config
        self.total_frames = total_frames

        # Internal storage: frame-by-frame tensors
        self._translation = torch.zeros(total_frames, config.translation_dim)
        self._rotation = torch.zeros(total_frames, 2) if config.has_rotation else None

    def set_frame(
        self,
        frame_idx: int,
        translation: List[float] | torch.Tensor | None = None,
        rotation: List[float] | torch.Tensor | None = None,
    ):
        """
        Set control parameters for a single frame.

        Args:
            frame_idx: Frame index to set
            translation: Translation vector (length = translation_dim)
            rotation: Rotation delta [pitch, yaw]
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise ValueError(f"frame_idx {frame_idx} out of range [0, {self.total_frames})")

        if translation is not None:
            if isinstance(translation, list):
                translation = torch.tensor(translation, dtype=torch.float32)
            self._translation[frame_idx] = translation

        if rotation is not None and self.config.has_rotation:
            if isinstance(rotation, list):
                rotation = torch.tensor(rotation, dtype=torch.float32)
            self._rotation[frame_idx] = rotation

    def set_segment(
        self,
        start_frame: int,
        end_frame: int,
        translation: List[float] | torch.Tensor | None = None,
        rotation: List[float] | torch.Tensor | None = None,
    ):
        """
        Set control parameters for a segment of frames.

        Args:
            start_frame: Start frame index (inclusive)
            end_frame: End frame index (exclusive)
            translation: Translation vector to apply to all frames
            rotation: Rotation delta to apply to all frames
        """
        for idx in range(start_frame, end_frame):
            self.set_frame(idx, translation, rotation)

    def get_segment(self, start_frame: int, num_frames: int) -> Dict[str, torch.Tensor]:
        """
        Extract control tensors for a segment (for streaming inference).

        Args:
            start_frame: Starting frame index
            num_frames: Number of frames to extract

        Returns:
            Dict with 'translation' and optionally 'rotation' tensors
        """
        end = min(start_frame + num_frames, self.total_frames)

        result = {
            "translation": self._translation[start_frame:end].clone(),
        }

        if self.config.has_rotation:
            result["rotation"] = self._rotation[start_frame:end].clone()

        return result

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Get all control tensors.

        Returns:
            Dict with 'translation' and optionally 'rotation' tensors
        """
        return self.get_segment(0, self.total_frames)


# ============================================================================
# Navigation pattern generation (for testing/benchmarking)
# ============================================================================


@dataclass
class NavigationPattern:
    """A named navigation pattern (e.g., forward+look_right)."""

    name: str
    translation_actions: List[str]
    rotation_actions: List[str]


class NavigationPatternGenerator:
    """Generates navigation patterns for testing and benchmarking."""

    def __init__(self, config: CameraControlConfig):
        self.config = config

    def build_standard_patterns(
        self,
        movement_weight: int = 5,
        rotation_weight: int = 5,
        combined_weight: int = 1,
    ) -> List[NavigationPattern]:
        """
        Build standard navigation patterns (forward, turn, forward+turn, etc.).

        Args:
            movement_weight: Repetition factor for pure movement patterns
            rotation_weight: Repetition factor for pure rotation patterns
            combined_weight: Repetition factor for movement+rotation combinations

        Returns:
            List of NavigationPattern objects
        """
        patterns = []

        # Pure translation patterns
        single_movements = ["forward", "left", "right"]  # excluding back for simplicity
        for action in single_movements:
            if action in self.config.translation.action_to_index:
                pattern = NavigationPattern(
                    name=action,
                    translation_actions=[action],
                    rotation_actions=[],
                )
                patterns.extend([pattern] * movement_weight)

        # Combined translation patterns
        combined_movements = [
            (["forward", "left"], "forward_left"),
            (["forward", "right"], "forward_right"),
        ]
        for actions, name in combined_movements:
            if all(a in self.config.translation.action_to_index for a in actions):
                pattern = NavigationPattern(
                    name=name,
                    translation_actions=actions,
                    rotation_actions=[],
                )
                patterns.extend([pattern] * movement_weight)

        # Pure rotation patterns
        rotation_actions = ["look_left", "look_right"]
        if self.config.has_rotation:
            for action in rotation_actions:
                if action in self.config.rotation.action_to_delta:
                    pattern = NavigationPattern(
                        name=action,
                        translation_actions=[],
                        rotation_actions=[action],
                    )
                    patterns.extend([pattern] * rotation_weight)

        # Movement + rotation combinations
        if self.config.has_rotation:
            all_movements = single_movements + ["forward_left", "forward_right"]
            for movement in all_movements:
                # Parse movement into action list
                if "_" in movement and movement in ["forward_left", "forward_right"]:
                    trans_actions = movement.split("_")[:2]
                else:
                    trans_actions = [movement]

                # Check availability
                if not all(a in self.config.translation.action_to_index for a in trans_actions):
                    continue

                for rotation in rotation_actions:
                    if rotation in self.config.rotation.action_to_delta:
                        pattern = NavigationPattern(
                            name=f"{movement}_{rotation}",
                            translation_actions=trans_actions,
                            rotation_actions=[rotation],
                        )
                        patterns.extend([pattern] * combined_weight)

        return patterns

    def pattern_to_tensors(
        self,
        pattern: NavigationPattern,
        num_frames: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a navigation pattern to control tensors.

        Args:
            pattern: Navigation pattern
            num_frames: Number of frames to generate

        Returns:
            Dict with 'translation' and 'rotation' tensors
        """
        result = {}

        # Translation tensor
        translation = torch.zeros(num_frames, self.config.translation_dim)
        for action in pattern.translation_actions:
            if action in self.config.translation.action_to_index:
                idx = self.config.translation.action_to_index[action]
                translation[:, idx] = 1.0
        result["translation"] = translation

        # Rotation tensor
        if self.config.has_rotation:
            rotation = torch.zeros(num_frames, 2)
            for action in pattern.rotation_actions:
                if action in self.config.rotation.action_to_delta:
                    delta = self.config.rotation.action_to_delta[action]
                    rotation[:, 0] = delta[0]  # pitch
                    rotation[:, 1] = delta[1]  # yaw
            result["rotation"] = rotation

        return result

    def generate_random_sequence(
        self,
        total_frames: int,
        num_frames_per_pattern: int = 4,
        segment_lengths: List[int] | None = None,
    ) -> CameraControlSequence:
        """
        Generate a random navigation sequence by combining patterns.

        Args:
            total_frames: Total number of frames (should be 4k+1)
            num_frames_per_pattern: Frames per individual pattern
            segment_lengths: Possible segment lengths to randomly choose from

        Returns:
            CameraControlSequence with randomized navigation
        """
        assert total_frames % 4 == 1, "total_frames should be 4k+1"

        if segment_lengths is None:
            segment_lengths = [12]

        # Build patterns
        patterns = self.build_standard_patterns()

        # Convert patterns to tensors
        pattern_tensors = [self.pattern_to_tensors(p, num_frames_per_pattern) for p in patterns]

        # Create sequence
        sequence = CameraControlSequence(self.config, total_frames)

        # Fill with random segments
        current_frame = 0
        import random

        while current_frame < total_frames:
            # Random segment length and pattern
            segment_len = random.choice(segment_lengths)
            pattern_idx = random.randint(0, len(pattern_tensors) - 1)
            pattern_tensor = pattern_tensors[pattern_idx]

            if current_frame == 0:
                # First frame
                sequence.set_frame(
                    0,
                    translation=pattern_tensor["translation"][0],
                    rotation=pattern_tensor.get("rotation", [None])[0]
                    if "rotation" in pattern_tensor
                    else None,
                )
                current_frame = 1
            else:
                # Subsequent frames
                remaining = total_frames - current_frame
                actual_len = min(segment_len, remaining)
                repeat_count = actual_len // 4

                if repeat_count > 0:
                    # Repeat pattern to fill segment
                    trans_repeated = pattern_tensor["translation"].repeat(repeat_count, 1)
                    sequence._translation[
                        current_frame : current_frame + actual_len
                    ] = trans_repeated

                    if self.config.has_rotation and "rotation" in pattern_tensor:
                        rot_repeated = pattern_tensor["rotation"].repeat(repeat_count, 1)
                        sequence._rotation[
                            current_frame : current_frame + actual_len
                        ] = rot_repeated

                current_frame += actual_len

        return sequence


# ============================================================================
# Public API
# ============================================================================


def generate_camera_navigation(
    total_frames: int = 57,
    num_frames_per_pattern: int = 4,
    config: CameraControlConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate random camera navigation sequence (for testing/benchmarking).

    This is the main entry point replacing the old Bench_actions_universal().

    Args:
        total_frames: Total number of frames (should be 4k+1)
        num_frames_per_pattern: Frames per individual pattern
        config: Camera control configuration (defaults to STANDARD_CAMERA_CONFIG)

    Returns:
        Dict with 'translation' and 'rotation' tensors
    """
    if config is None:
        config = STANDARD_CAMERA_CONFIG

    generator = NavigationPatternGenerator(config)
    sequence = generator.generate_random_sequence(
        total_frames=total_frames,
        num_frames_per_pattern=num_frames_per_pattern,
    )

    return sequence.to_tensors()
