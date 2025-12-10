"""
Camera pose management and transformation utilities.

This module provides tools for working with camera poses (position + orientation)
and computing transformations between them.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class CameraPose:
    """
    Represents a camera pose with position and rotation.

    Attributes:
        position: 3D position [x, y, z] as tensor of shape (3,)
        rotation: Rotation as quaternion [w, x, y, z] as tensor of shape (4,)
    """

    position: torch.Tensor  # shape: (3,)
    rotation: torch.Tensor  # shape: (4,) - quaternion [w, x, y, z]

    def __post_init__(self):
        """Validate and normalize the pose."""
        assert self.position.shape == (
            3,
        ), f"Position must be shape (3,), got {self.position.shape}"
        assert self.rotation.shape == (
            4,
        ), f"Rotation must be shape (4,), got {self.rotation.shape}"
        # Normalize quaternion
        self.rotation = F.normalize(self.rotation, dim=0)

    @classmethod
    def from_position_euler(
        cls,
        position: torch.Tensor | List[float],
        euler_angles: torch.Tensor | List[float],
        order: str = "xyz",
    ) -> "CameraPose":
        """
        Create a pose from position and Euler angles.

        Args:
            position: [x, y, z]
            euler_angles: [roll, pitch, yaw] in radians
            order: Rotation order (default: "xyz")

        Returns:
            CameraPose instance
        """
        if isinstance(position, list):
            position = torch.tensor(position, dtype=torch.float32)
        if isinstance(euler_angles, list):
            euler_angles = torch.tensor(euler_angles, dtype=torch.float32)

        quaternion = euler_to_quaternion(euler_angles, order)
        return cls(position=position, rotation=quaternion)

    @classmethod
    def from_position_matrix(
        cls, position: torch.Tensor | List[float], rotation_matrix: torch.Tensor
    ) -> "CameraPose":
        """
        Create a pose from position and rotation matrix.

        Args:
            position: [x, y, z]
            rotation_matrix: 3x3 rotation matrix

        Returns:
            CameraPose instance
        """
        if isinstance(position, list):
            position = torch.tensor(position, dtype=torch.float32)

        quaternion = matrix_to_quaternion(rotation_matrix)
        return cls(position=position, rotation=quaternion)

    @classmethod
    def identity(cls, device: Optional[torch.device] = None) -> "CameraPose":
        """Create an identity pose (origin with no rotation)."""
        return cls(
            position=torch.zeros(3, device=device),
            rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        )

    def to_euler(self, order: str = "xyz") -> torch.Tensor:
        """Convert rotation to Euler angles [roll, pitch, yaw]."""
        return quaternion_to_euler(self.rotation, order)

    def to_matrix(self) -> torch.Tensor:
        """Convert rotation to 3x3 rotation matrix."""
        return quaternion_to_matrix(self.rotation)

    def clone(self) -> "CameraPose":
        """Create a deep copy of this pose."""
        return CameraPose(position=self.position.clone(), rotation=self.rotation.clone())


@dataclass
class PoseTransform:
    """
    Represents a transformation between poses.

    Attributes:
        translation: 3D translation vector
        rotation: Rotation as quaternion [w, x, y, z]
    """

    translation: torch.Tensor  # shape: (3,)
    rotation: torch.Tensor  # shape: (4,) - quaternion

    def __post_init__(self):
        """Validate and normalize."""
        assert self.translation.shape == (
            3,
        ), f"Translation must be shape (3,), got {self.translation.shape}"
        assert self.rotation.shape == (
            4,
        ), f"Rotation must be shape (4,), got {self.rotation.shape}"
        self.rotation = F.normalize(self.rotation, dim=0)


# ============================================================================
# Core Pose Operations
# ============================================================================


def apply_transforms(initial_pose: CameraPose, transforms: List[PoseTransform]) -> CameraPose:
    """
    Apply multiple transforms sequentially to get the final pose.

    This computes: final_pose = initial_pose ∘ T₁ ∘ T₂ ∘ ... ∘ Tₙ

    Transform semantics: Each transform specifies translation in the LOCAL frame
    before rotation, then applies rotation.

    Args:
        initial_pose: Starting camera pose
        transforms: List of transforms to apply in order

    Returns:
        Final camera pose after all transforms

    Example:
        >>> pose = CameraPose.identity()
        >>> t1 = PoseTransform(torch.tensor([1.0, 0, 0]), torch.tensor([1, 0, 0, 0]))
        >>> t2 = PoseTransform(torch.tensor([0, 1.0, 0]), torch.tensor([1, 0, 0, 0]))
        >>> final = apply_transforms(pose, [t1, t2])
        >>> # final.position will be approximately [1, 1, 0]
    """
    current_pose = initial_pose.clone()

    if transforms:
        target_device = transforms[0].translation.device
        current_pose.position = current_pose.position.to(target_device)
        current_pose.rotation = current_pose.rotation.to(target_device)

    for transform in transforms:
        rotated_translation = rotate_vector_by_quaternion(
            transform.translation, current_pose.rotation
        )
        current_pose.position = current_pose.position + rotated_translation
        current_pose.rotation = quaternion_multiply(current_pose.rotation, transform.rotation)

    return current_pose


def compute_transform(pose_a: CameraPose, pose_b: CameraPose) -> PoseTransform:
    """
    Compute the transform that takes pose_a to pose_b.

    Returns a PoseTransform such that: pose_b = pose_a ∘ transform

    The transform semantics match apply_transforms: translation is in pose_a's
    local frame, applied before rotation.

    Args:
        pose_a: Starting pose
        pose_b: Target pose

    Returns:
        PoseTransform that moves from pose_a to pose_b

    Example:
        >>> pose_a = CameraPose.identity()
        >>> pose_b = CameraPose.from_position_euler([1, 2, 3], [0, 0, 0])
        >>> transform = compute_transform(pose_a, pose_b)
        >>> # transform can be applied to pose_a to get pose_b
        >>> result = apply_transforms(pose_a, [transform])
        >>> # result should match pose_b
    """
    # Compute relative rotation: q_rel = q_a^{-1} * q_b
    q_a_inv = quaternion_conjugate(pose_a.rotation)
    relative_rotation = quaternion_multiply(q_a_inv, pose_b.rotation)

    # Compute relative translation in pose_a's local frame:
    # Since apply_transforms does: p_new = p_old + rotate(t, q_old)
    # We need: t = rotate^{-1}(p_b - p_a, q_a)
    position_diff = pose_b.position - pose_a.position
    relative_translation = rotate_vector_by_quaternion(position_diff, q_a_inv)

    return PoseTransform(translation=relative_translation, rotation=relative_rotation)


def find_closest_pose(
    reference_pose: CameraPose,
    candidate_poses: List[CameraPose],
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> Tuple[int, CameraPose, float]:
    """
    Find the closest pose from candidates to the reference pose.

    Distance is computed as a weighted sum of position and rotation distances:
        distance = position_weight * ||p_ref - p_i||² + rotation_weight * angle²

    Args:
        reference_pose: The pose to match
        candidate_poses: List of candidate poses
        position_weight: Weight for position distance (default: 1.0)
        rotation_weight: Weight for rotation distance (default: 1.0)

    Returns:
        Tuple of (index, closest_pose, distance)

    Example:
        >>> ref = CameraPose.identity()
        >>> candidates = [
        ...     CameraPose.from_position_euler([0.1, 0, 0], [0, 0, 0]),
        ...     CameraPose.from_position_euler([5.0, 0, 0], [0, 0, 0]),
        ... ]
        >>> idx, closest, dist = find_closest_pose(ref, candidates)
        >>> # idx will be 0 (first pose is closer)
    """
    if not candidate_poses:
        raise ValueError("candidate_poses cannot be empty")

    min_distance = float("inf")
    closest_idx = 0
    closest_pose = candidate_poses[0]

    for i, candidate in enumerate(candidate_poses):
        # Position distance (squared Euclidean)
        pos_dist = torch.sum((reference_pose.position - candidate.position) ** 2).item()

        # Rotation distance (angular distance)
        rot_dist = quaternion_angular_distance(reference_pose.rotation, candidate.rotation)

        # Combined distance
        distance = position_weight * pos_dist + rotation_weight * rot_dist

        if distance < min_distance:
            min_distance = distance
            closest_idx = i
            closest_pose = candidate

    return closest_idx, closest_pose, min_distance


# ============================================================================
# Quaternion Math Utilities
# ============================================================================


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Multiply two quaternions: q1 * q2.

    Args:
        q1, q2: Quaternions as [w, x, y, z]
        normalize: Whether to normalize the result (default: True)

    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = torch.stack([w, x, y, z])
    return F.normalize(result, dim=0) if normalize else result


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion conjugate (inverse for unit quaternions).

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Conjugate [w, -x, -y, -z]
    """
    return torch.stack([q[0], -q[1], -q[2], -q[3]])


def rotate_vector_by_quaternion(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate a 3D vector by a quaternion.

    Uses: v' = q * v * q^{-1} (treating v as pure quaternion [0, v])

    Args:
        v: 3D vector to rotate
        q: Rotation quaternion [w, x, y, z]

    Returns:
        Rotated 3D vector
    """
    v_quat = torch.cat([torch.zeros(1, device=v.device, dtype=v.dtype), v])

    q_conj = quaternion_conjugate(q)
    temp = quaternion_multiply(q, v_quat, normalize=False)
    result = quaternion_multiply(temp, q_conj, normalize=False)

    return result[1:]


def quaternion_angular_distance(q1: torch.Tensor, q2: torch.Tensor) -> float:
    """
    Compute angular distance between two quaternions.

    Args:
        q1, q2: Quaternions [w, x, y, z]

    Returns:
        Angular distance in radians squared (for efficiency)
    """
    # Dot product (taking absolute value to handle double cover)
    dot = torch.abs(torch.dot(q1, q2))
    dot = torch.clamp(dot, -1.0, 1.0)

    # Angular distance: 2 * arccos(|dot|)
    angle = 2.0 * torch.acos(dot)
    return (angle**2).item()


# ============================================================================
# Conversion Utilities
# ============================================================================


def euler_to_quaternion(euler: torch.Tensor, order: str = "xyz") -> torch.Tensor:
    """
    Convert Euler angles to quaternion.

    Args:
        euler: [roll, pitch, yaw] in radians
        order: Rotation order (default: "xyz")

    Returns:
        Quaternion [w, x, y, z]
    """
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    if order.lower() == "xyz":
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
    else:
        raise NotImplementedError(f"Order '{order}' not implemented")

    return F.normalize(torch.stack([w, x, y, z]), dim=0)


def quaternion_to_euler(q: torch.Tensor, order: str = "xyz") -> torch.Tensor:
    """
    Convert quaternion to Euler angles.

    Args:
        q: Quaternion [w, x, y, z]
        order: Rotation order (default: "xyz")

    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    if order.lower() == "xyz":
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw])
    else:
        raise NotImplementedError(f"Order '{order}' not implemented")


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)

    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)

    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    return torch.stack(
        [torch.stack([r00, r01, r02]), torch.stack([r10, r11, r12]), torch.stack([r20, r21, r22])]
    )


def matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to quaternion.

    Args:
        m: 3x3 rotation matrix

    Returns:
        Quaternion [w, x, y, z]
    """
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return F.normalize(torch.stack([w, x, y, z]), dim=0)
