import typing
from typing import TYPE_CHECKING, List, Optional

import torch
from torch import nn

from grotto.modeling.causal_model import CausalWanModel
from grotto.modeling.modular_action import ActionContext
from grotto.types import ConditionalInputs

if TYPE_CHECKING:
    from .kv_cache import DualPlaneKVCache


class WanDiffusionPredictor(nn.Module):
    def __init__(
        self,
        num_frame_per_block,
        model_config_path: str = "",
        timestep_shift: float = 5.0,
        is_causal: bool = True,
    ):
        super().__init__()

        self.num_frame_per_block = num_frame_per_block

        self.model_config = CausalWanModel.load_config(model_config_path)
        self.model = typing.cast(CausalWanModel, CausalWanModel.from_config(self.model_config))  # type: ignore
        self.model.eval()

        self.uniform_timestep = not is_causal
        self.timestep_shift = timestep_shift

        self.seq_len = 15 * 880  # 32760  # [1, 15, 16, 60, 104]

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_inputs: ConditionalInputs,
        timestep: torch.Tensor,
        kv_cache: List,
        kv_cache_mouse: Optional[List["DualPlaneKVCache"]] = None,
        kv_cache_keyboard: Optional[List["DualPlaneKVCache"]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        cache_mode: str = "read_write",
    ) -> torch.Tensor:
        assert noisy_image_or_video.shape[1] == 16

        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        action_context = None
        if (
            conditional_inputs.rotation_cond is not None
            or conditional_inputs.translation_cond is not None
        ):
            action_context = ActionContext(
                rotation_cond=conditional_inputs.rotation_cond,
                translation_cond=conditional_inputs.translation_cond,
                num_frame_per_block=self.num_frame_per_block,
            )

        flow_pred = self.model(
            noisy_image_or_video.to(self.model.dtype),
            timesteps=input_timestep,
            visual_context=conditional_inputs.visual_context,
            cond_concat=conditional_inputs.cond_concat,
            action_context=action_context,
            kv_cache=kv_cache,
            kv_cache_mouse=kv_cache_mouse,
            kv_cache_keyboard=kv_cache_keyboard,
            current_start=current_start,
            cache_mode=cache_mode,
        )

        return flow_pred
