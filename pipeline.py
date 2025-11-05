from .models.wan_models import WanDiffusionPredictor

from omegaconf import OmegaConf
import torch


class InteractiveGamePipeline:
    def __init__(self, config_path: str,
                 checkpoint_path: str,
                 img_path: str,
                 output_folder: str,
                 num_output_frames: int,
                 seed: int,
                 pretrained_model_path: str,
                 enable_profile: bool,
                 vae_compile_mode: str):
        self.enable_profile = False

        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self.config = OmegaConf.load(config_path)

    def _init_model(self):
        predictor = WanDiffusionPredictor(
            model_config_path=self.config.model_config_path,
            timestep_shift=self.config.timestep_shift,
            is_causal=True)
        
        