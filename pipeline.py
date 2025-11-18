from omegaconf import OmegaConf
import torch
import os
from enum import Enum
import logging

from .models.wan_models import WanDiffusionPredictor
from .models.vae_wrapper import VaeDecoderWrapper

class VAECompileMode(str, Enum):
    AUTO = "auto"
    FORCE = "force"
    NONE = "none"

class InteractiveGamePipeline:
    def __init__(self, config_path: str,
                 checkpoint_path: str,
                 img_path: str,
                 output_folder: str,
                 num_output_frames: int,
                 seed: int,
                 pretrained_model_path: str,
                 enable_profile: bool,
                 vae_compile_mode: VAECompileMode):
        self.enable_profile = False

        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self.config = OmegaConf.load(config_path)

    def _init_model(self, pretrained_model_path: str, compile_mode: VAECompileMode):
        predictor = WanDiffusionPredictor(
            model_config_path=self.config.model_config_path,
            is_causal=True)
        
        vae_decoder = VaeDecoderWrapper()

        compiled_model_path = os.path.join(pretrained_model_path, "compiled_vae_decoder.pt")
        vae_state_dict = torch.load(os.path.join(pretrained_model_path, "Wan2.1_VAE.pth"), map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value

        vae_decoder.load_state_dict(decoder_state_dict)
        vae_decoder.to(device=self.device, dtype=torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()

        if compile_mode == VAECompileMode.NONE:
            logging.info("VAE decoder compilation skipped as per user request.")
        elif compile_mode == VAECompileMode.FORCE:
            logging.info("Forcing VAE decoder compilation...")
            vae_decoder = torch.compile(vae_decoder, mode="max-autotune")
            torch.save(vae_decoder, compiled_model_path)
            logging.info(f"Compiled VAE decoder saved to {compiled_model_path}")
        elif compile_mode == VAECompileMode.AUTO:
            if os.path.exists(compiled_model_path):
                logging.info(f"Loading compiled VAE decoder from {compiled_model_path}...")
                vae_decoder = torch.load(compiled_model_path, map_location=self.device)
            else:
                logging.info("Compiling VAE decoder...")
                vae_decoder = torch.compile(vae_decoder, mode="max-autotune")
                torch.save(vae_decoder, compiled_model_path)
                logging.info(f"Compiled VAE decoder saved to {compiled_model_path}")

        

