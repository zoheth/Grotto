"""
Demo of camera pose operations.

This example shows how to use the camera pose utilities for:
1. Applying multiple transforms to get final pose
2. Computing transform between two poses
3. Finding closest pose from a set of candidates
"""

import torch

from grotto.camera_pose import (
    CameraPose,
    PoseTransform,
    apply_transforms,
    compute_transform,
    find_closest_pose,
)


def demo_apply_transforms():
    """Demo: Apply multiple transforms to get final pose."""
    print("=" * 60)
    print("Demo 1: Apply Multiple Transforms")
    print("=" * 60)

    # Start at origin
    initial_pose = CameraPose.identity()
    print(f"Initial pose: position={initial_pose.position}, rotation={initial_pose.rotation}")

    # Create transforms: move forward 2 units, then rotate 90° around Y, then move right 1 unit
    transforms = [
        # Move forward (along X)
        PoseTransform(
            translation=torch.tensor([2.0, 0.0, 0.0]),
            rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]),  # No rotation
        ),
        # Rotate 90° around Y axis (yaw)
        PoseTransform(
            translation=torch.tensor([0.0, 0.0, 0.0]),
            rotation=torch.tensor([0.7071, 0.0, 0.7071, 0.0]),  # 90° around Y
        ),
        # Move forward in new direction
        PoseTransform(
            translation=torch.tensor([1.0, 0.0, 0.0]), rotation=torch.tensor([1.0, 0.0, 0.0, 0.0])
        ),
    ]

    # Apply all transforms
    final_pose = apply_transforms(initial_pose, transforms)
    print(f"Final pose: position={final_pose.position}")
    print(f"Final euler angles: {final_pose.to_euler()}")
    print()


def demo_compute_transform():
    """Demo: Compute transform between two poses."""
    print("=" * 60)
    print("Demo 2: Compute Transform Between Poses")
    print("=" * 60)

    # Create two poses
    pose_a = CameraPose.from_position_euler(position=[0.0, 0.0, 0.0], euler_angles=[0.0, 0.0, 0.0])
    pose_b = CameraPose.from_position_euler(
        position=[3.0, 1.0, 2.0],
        euler_angles=[0.0, 1.57, 0.0],  # 90° yaw
    )

    print(f"Pose A: position={pose_a.position}")
    print(f"Pose B: position={pose_b.position}, euler={pose_b.to_euler()}")

    # Compute transform
    transform = compute_transform(pose_a, pose_b)
    print(f"Transform: translation={transform.translation}")

    # Verify: apply transform to pose_a should give pose_b
    reconstructed = apply_transforms(pose_a, [transform])
    print(f"Reconstructed pose: position={reconstructed.position}")
    print(f"Position error: {torch.norm(reconstructed.position - pose_b.position).item():.6f}")
    print()


def demo_find_closest():
    """Demo: Find closest pose from candidates."""
    print("=" * 60)
    print("Demo 3: Find Closest Pose")
    print("=" * 60)

    # Reference pose
    reference = CameraPose.from_position_euler(
        position=[1.0, 0.0, 0.0], euler_angles=[0.0, 0.0, 0.0]
    )

    # Candidate poses
    candidates = [
        CameraPose.from_position_euler([0.9, 0.1, 0.1], [0.0, 0.1, 0.0]),  # Close
        CameraPose.from_position_euler([5.0, 2.0, 3.0], [1.0, 1.0, 1.0]),  # Far
        CameraPose.from_position_euler([1.1, 0.0, 0.1], [0.0, 0.0, 0.1]),  # Very close
        CameraPose.from_position_euler([10.0, 10.0, 10.0], [0.0, 0.0, 0.0]),  # Very far
    ]

    print(f"Reference pose: position={reference.position}")
    print(f"Number of candidates: {len(candidates)}")

    # Find closest
    idx, closest_pose, distance = find_closest_pose(reference, candidates)

    print(f"Closest pose index: {idx}")
    print(f"Closest pose position: {closest_pose.position}")
    print(f"Distance: {distance:.6f}")

    # Show all distances
    print("\nAll candidate distances:")
    for i, candidate in enumerate(candidates):
        pos_dist = torch.norm(reference.position - candidate.position).item()
        print(f"  Candidate {i}: position distance = {pos_dist:.4f}")
    print()


def demo_integration_with_camera_control():
    """Demo: How to integrate with existing camera control system."""
    print("=" * 60)
    print("Demo 4: Integration with Camera Control")
    print("=" * 60)

    # Suppose you have camera control deltas (pitch, yaw) from CameraControlSequence
    # and you want to convert them to pose transforms

    # Example: camera rotates 10° right (yaw), then moves forward
    yaw_delta = 0.174  # ~10° in radians
    # pitch_delta = 0.0

    # Current pose
    current_pose = CameraPose.identity()

    # Convert rotation delta to transform
    rotation_transform = PoseTransform(
        translation=torch.zeros(3),
        rotation=torch.tensor(
            [
                torch.cos(torch.tensor(yaw_delta / 2)),
                0.0,
                torch.sin(torch.tensor(yaw_delta / 2)),
                0.0,
            ]
        ),
    )

    # Movement transform (forward = positive X in camera frame)
    movement_transform = PoseTransform(
        translation=torch.tensor([1.0, 0.0, 0.0]), rotation=torch.tensor([1.0, 0.0, 0.0, 0.0])
    )

    # Apply transforms
    final_pose = apply_transforms(current_pose, [rotation_transform, movement_transform])

    print("After rotation + movement:")
    print(f"  Position: {final_pose.position}")
    print(f"  Euler angles: {final_pose.to_euler()}")
    print()


if __name__ == "__main__":
    demo_apply_transforms()
    demo_compute_transform()
    demo_find_closest()
    demo_integration_with_camera_control()

    print("=" * 60)
    print("All demos completed!")
    print("=" * 60)
