from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import torch

class ActionDict(Dict[str, torch.Tensor]):
    """Type-safe action dictionary."""

    def __init__(self, **kwargs):
        """
        Initialize action dictionary.

        Valid keys depend on the game mode:
        - universal/gta_drive: 'mouse', 'keyboard'
        - templerun: 'keyboard'
        """
        super().__init__(**kwargs)

    def validate(self, mode: str) -> None:
        """Validate that the action dict has required keys for the mode."""
        if mode == 'templerun':
            required = {'keyboard'}
        else:
            required = {'mouse', 'keyboard'}

        if not required.issubset(self.keys()):
            raise ValueError(f"Missing required keys for mode {mode}: {required - self.keys()}")
