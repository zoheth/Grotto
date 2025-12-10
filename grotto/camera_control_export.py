"""
Camera control export utilities for cross-project compatibility.

This module provides functions to save camera control sequences from Grotto
and convert them to Matrix-Game format for comparison.
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from grotto.types import CameraControlTensors


def save_camera_control(
    camera_control: CameraControlTensors, save_path: Union[str, Path], format: str = "pt"
) -> None:
    """
    Save camera control tensors to file.

    Args:
        camera_control: CameraControlTensors object containing rotation and translation
        save_path: Path to save the file
        format: Format to save ('pt' for PyTorch, 'npz' for NumPy)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "pt":
        data = {
            "translation": camera_control.translation,
            "rotation": camera_control.rotation,
        }
        torch.save(data, save_path)
    elif format == "npz":
        data = {
            "translation": camera_control.translation.cpu().numpy(),
            "rotation": camera_control.rotation.cpu().numpy()
            if camera_control.rotation is not None
            else None,
        }
        np.savez(save_path, **data)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'pt' or 'npz'.")

    print(f"Camera control saved to {save_path}")


def convert_to_matrix_game_format(
    camera_control: CameraControlTensors, remove_batch_dim: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Convert Grotto camera control to Matrix-Game format.

    Grotto format:
        - translation: [B, num_frames, 4] or [num_frames, 4] - WASD movement
        - rotation: [B, num_frames, 2] or [num_frames, 2] - [pitch, yaw]

    Matrix-Game format:
        - keyboard_condition: [num_frames, 4] - [forward, back, left, right]
        - mouse_condition: [num_frames, 2] - [pitch, yaw]

    Args:
        camera_control: CameraControlTensors from Grotto
        remove_batch_dim: If True, remove batch dimension (default for Matrix-Game)

    Returns:
        Dictionary with 'keyboard_condition' and 'mouse_condition'
    """
    translation = camera_control.translation
    rotation = camera_control.rotation

    # Remove batch dimension if needed
    if remove_batch_dim and translation.dim() == 3:
        translation = translation.squeeze(0)
    if remove_batch_dim and rotation is not None and rotation.dim() == 3:
        rotation = rotation.squeeze(0)

    # The fields map directly:
    # Grotto translation [forward, back, left, right] -> Matrix-Game keyboard_condition
    # Grotto rotation [pitch, yaw] -> Matrix-Game mouse_condition
    matrix_game_format = {
        "keyboard_condition": translation,
        "mouse_condition": rotation
        if rotation is not None
        else torch.zeros_like(translation[:, :2]),
    }

    return matrix_game_format


def save_for_matrix_game(camera_control: CameraControlTensors, save_path: Union[str, Path]) -> None:
    """
    Save camera control in Matrix-Game compatible format.

    Args:
        camera_control: CameraControlTensors from Grotto
        save_path: Path to save the file (.pt format)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    matrix_game_data = convert_to_matrix_game_format(camera_control)
    torch.save(matrix_game_data, save_path)

    print(f"Camera control saved in Matrix-Game format to {save_path}")
    print(f"  - keyboard_condition shape: {matrix_game_data['keyboard_condition'].shape}")
    print(f"  - mouse_condition shape: {matrix_game_data['mouse_condition'].shape}")


def load_camera_control(load_path: Union[str, Path], device: str = "cpu") -> CameraControlTensors:
    """
    Load camera control from file.

    Args:
        load_path: Path to the saved file
        device: Device to load tensors to

    Returns:
        CameraControlTensors object
    """
    load_path = Path(load_path)

    if load_path.suffix == ".pt":
        data = torch.load(load_path, map_location=device)
        return CameraControlTensors(
            translation=data["translation"],
            rotation=data.get("rotation"),
        )
    elif load_path.suffix == ".npz":
        data = np.load(load_path)
        return CameraControlTensors(
            translation=torch.from_numpy(data["translation"]).to(device),
            rotation=torch.from_numpy(data["rotation"]).to(device)
            if "rotation" in data and data["rotation"] is not None
            else None,
        )
    else:
        raise ValueError(f"Unsupported file format: {load_path.suffix}")
