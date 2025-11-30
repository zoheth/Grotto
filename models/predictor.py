from dataclasses import dataclass
import typing
from typing import Optional, List, Dict, TYPE_CHECKING

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from .causal_model import CausalWanModel
from ..scheduler import FlowMatchScheduler
from .modular_action import ActionContext

if TYPE_CHECKING:
    from .paged_cache import PagedCache
    from .ring_buffer_cache import RingBufferActionCache

class WanDiffusionPredictor(nn.Module):
    def __init__(self,
                 num_frame_per_block,
                 model_config_path:str = "",
                 timestep_shift: float = 5.0,
                 is_causal: bool = True):
        super().__init__()

        self.num_frame_per_block = num_frame_per_block

        self.model_config, _ = CausalWanModel.load_config(model_config_path)
        self.model= typing.cast(CausalWanModel, CausalWanModel.from_config(self.model_config))
        self.model.eval()

        self.uniform_timestep = not is_causal
        self.timestep_shift = timestep_shift

        self.seq_len = 15 * 880 # 32760  # [1, 15, 16, 60, 104]

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: List["PagedCache"],
        kv_cache_mouse: Optional[List["RingBufferActionCache"]] = None,
        kv_cache_keyboard: Optional[List["RingBufferActionCache"]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None
    ) -> torch.Tensor:
        assert noisy_image_or_video.shape[1] == 16
        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        mouse_cond = conditional_dict.get("mouse_cond")
        keyboard_cond = conditional_dict.get("keyboard_cond")

        action_context = None
        if mouse_cond is not None or keyboard_cond is not None:
            action_context = ActionContext(
                mouse_cond=mouse_cond,
                keyboard_cond=keyboard_cond,
                num_frame_per_block=self.num_frame_per_block
            )

        flow_pred = self.model(
            noisy_image_or_video.to(self.model.dtype),
            timesteps = input_timestep,
            visual_context = conditional_dict.get('visual_context'),
            action_context = action_context,
            kv_cache = kv_cache,
            kv_cache_mouse = kv_cache_mouse,
            kv_cache_keyboard = kv_cache_keyboard,
            current_start = current_start
        )

        return flow_pred