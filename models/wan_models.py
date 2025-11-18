import typing

import torch
from torch import nn

from .causal_model import CausalWanModel
from ..scheduler import FlowMatchScheduler



class WanDiffusionPredictor(nn.Module):
    def __init__(self,
                 model_config_path:str = "",
                 num_frame_per_block: int = 1,
                 is_causal: bool = True):
        super().__init__()

        self.model_config, _ = CausalWanModel.load_config(model_config_path)
        self.model= typing.cast(CausalWanModel, CausalWanModel.from_config(self.model_config))
        self.model.eval()

        self.uniform_timestep = not is_causal

        self.seq_len = 15 * 880 # 32760  # [1, 15, 16, 60, 104]
