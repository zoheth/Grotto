"""
Type definitions for Grotto.

This module contains shared type definitions to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ConditionalInputs:
    """
    Encapsulates all conditional inputs for video generation.

    Attributes:
        cond_concat: Concatenated conditioning (mask + initial frame latents) [B, C, F, H, W]
        visual_context: CLIP visual features [B, SeqLen, Dim]
        rotation_cond: Camera rotation conditioning [B, SeqLen, RotDim]
        translation_cond: Camera translation conditioning [B, SeqLen, TransDim]
    """

    cond_concat: torch.Tensor
    visual_context: torch.Tensor
    rotation_cond: Optional[torch.Tensor] = None
    translation_cond: Optional[torch.Tensor] = None

    def to(self, device: torch.device, dtype: torch.dtype) -> "ConditionalInputs":
        """
        Move all tensors to the specified device and dtype.

        Args:
            device: Target device
            dtype: Target dtype

        Returns:
            New ConditionalInputs instance with tensors on the target device/dtype
        """
        return ConditionalInputs(
            cond_concat=self.cond_concat.to(device=device, dtype=dtype),
            visual_context=self.visual_context.to(device=device, dtype=dtype),
            rotation_cond=self.rotation_cond.to(device=device, dtype=dtype)
            if self.rotation_cond is not None
            else None,
            translation_cond=self.translation_cond.to(device=device, dtype=dtype)
            if self.translation_cond is not None
            else None,
        )
