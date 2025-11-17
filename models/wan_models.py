import typing

import torch
from torch import nn

from .causal_model import CausalWanModel
from ..scheduler import FlowMatchScheduler



class WanDiffusionPredictor(nn.Module):
    def __init__(self,
                 model_config_path:str = "",
                 timestep_shift: float = 5.0,
                 is_causal: bool = True):
        super().__init__()

        self.model_config, _ = CausalWanModel.load_config(model_config_path)
        self.model= typing.cast(CausalWanModel, CausalWanModel.from_config(self.model_config))
        self.model.eval()

        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_training_timesteps(num_steps=1000)

        self.seq_len = 15 * 880 # 32760  # [1, 15, 16, 60, 104]
