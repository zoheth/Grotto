from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import nn

if TYPE_CHECKING:
    pass


class ActionInjector(nn.Module, ABC):
    """Base class for action condition injectors."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        spatial_shape: Tuple[int, int],
        temporal_shape: int,
        is_causal: bool = False,
        kv_cache=None,
        start_frame: int = 0,
        num_frame_per_block: int = 1,
    ) -> torch.Tensor:
        pass
