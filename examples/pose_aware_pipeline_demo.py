"""
Demo of pose-aware pipeline with temporal consistency.

This shows how the pipeline tracks poses and adjusts actions to maintain
consistency when reusing cached features.
"""

import torch

from grotto.camera_pose import CameraPose
from grotto.pipeline.pose_tracker import PoseActionAdjuster, PoseTracker


def demo_pose_tracking():
    """Demonstrate basic pose tracking."""
    print("=" * 60)
    print("Demo: Pose Tracking")
    print("=" * 60)

    # Initialize tracker
    tracker = PoseTracker(initial_pose=CameraPose.identity())

    # Simulate a sequence of actions
    # Action format: [forward, back, left, right]
    actions_translation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # Forward
            [1.0, 0.0, 0.0, 0.0],  # Forward
            [0.0, 0.0, 0.0, 1.0],  # Right
            [1.0, 0.0, 0.0, 0.0],  # Forward
        ]
    )

    # Rotation format: [pitch, yaw]
    actions_rotation = torch.tensor(
        [
            [0.0, 0.0],  # No rotation
            [0.0, 0.1],  # Turn slightly right
            [0.0, 0.1],  # Turn slightly right
            [0.0, 0.0],  # No rotation
        ]
    )

    print("Applying action sequence...")
    tracker.apply_action_sequence(
        translations=actions_translation, rotations=actions_rotation, start_action_index=0
    )

    print(f"\nHistory has {len(tracker.history)} entries:")
    for _, entry in enumerate(tracker.history[:5]):  # Show first 5
        print(f"  Action {entry.action_index}: position={entry.pose.position.numpy()}")

    print(f"\nCurrent pose: {tracker.current_pose.position.numpy()}")
    print()


def demo_action_adjustment():
    """Demonstrate action adjustment for consistency."""
    print("=" * 60)
    print("Demo: Action Adjustment")
    print("=" * 60)

    # Setup: simulate a scenario where we've generated some frames
    tracker = PoseTracker()

    # Generate history (e.g., moved forward 3 times)
    for i in range(3):
        tracker.apply_action(
            translation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            rotation=torch.zeros(2),
            action_index=i,
        )

    print(f"History length: {len(tracker.history)}")
    print(f"Current position: {tracker.current_pose.position.numpy()}")

    # Now simulate a new action that happens to lead back to action index 1
    adjuster = PoseActionAdjuster(tracker)

    # New action that approximately returns to position at action 1
    new_action_translation = torch.tensor([0.0, 1.0, 0.0, 1.0])  # Back + left
    new_action_rotation = torch.zeros(2)

    print("\nChecking if adjustment needed...")
    (
        adjusted_trans,
        adjusted_rot,
        truncate_idx,
    ) = adjuster.adjust_first_action_and_get_cache_truncation(
        new_action_translation, new_action_rotation, current_action_index=3
    )

    if truncate_idx is not None:
        print(f"✓ Adjustment triggered! Truncate cache after action {truncate_idx}")
        print(f"  Original translation: {new_action_translation.numpy()}")
        print(f"  Adjusted translation: {adjusted_trans.numpy()}")
    else:
        print("✗ No adjustment needed (poses not close enough)")

    print()


def demo_integration_concept():
    """Show how this integrates with pipeline."""
    print("=" * 60)
    print("Demo: Pipeline Integration Concept")
    print("=" * 60)

    print(
        """
Pipeline Integration Flow:

1. Initialize PoseTracker with starting pose
2. For each block iteration:
   a. Extract first action of new block
   b. Use PoseActionAdjuster to check if adjustment needed
   c. If adjustment triggered:
      - Replace first action with adjusted action
      - Truncate KV cache to match historical pose
      - Truncate pose history
   d. Generate frames with (potentially adjusted) actions
   e. Update pose tracker with actual actions used

Key Benefits:
- Maintains temporal consistency when cache is reused
- Prevents drift by anchoring to historical poses
- Seamless integration with existing pipeline

Example Usage:
```python
from grotto.pipeline.pose_aware_pipeline import PoseAwarePipeline

pipeline = PoseAwarePipeline(
    config=config,
    predictor=predictor,
    vae_decoder=vae_decoder,
    initial_pose=CameraPose.identity(),
    enable_pose_adjustment=True  # Enable consistency checks
)

videos = pipeline.inference(noise, conditional_inputs)
```
    """
    )


if __name__ == "__main__":
    demo_pose_tracking()
    demo_action_adjustment()
    demo_integration_concept()
