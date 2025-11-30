import os
import torch
from enum import Enum
import logging
import numpy as np

import PIL.Image
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torchvision.transforms import v2
from .models.predictor import WanDiffusionPredictor
from .pipeline import PipelineConfig, BatchCausalInferencePipeline
from .models.vae_wrapper import VaeDecoderWrapper
from .models.vae_wrapper import create_wan_encoder
from .conditions import Bench_actions_universal

class VAECompileMode(str, Enum):
    AUTO = "auto"
    FORCE = "force"
    NONE = "none"

class VideoGenerator:
    def __init__(self, config_path, checkpoint_path, vae_dir, device="cuda"):
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16
        self.config:PipelineConfig = PipelineConfig.load(config_path)

        self._init_models(checkpoint_path, vae_dir)

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_models(self, checkpoint_path, vae_dir):
        predictor = WanDiffusionPredictor(
            num_frame_per_block=self.config.inference.num_frame_per_block,
            model_config_path=self.config.model_config_path,
            timestep_shift=self.config.inference.timestep_shift
        )

        state_dict = load_file(checkpoint_path)
        predictor.load_state_dict(state_dict)

        vae_decoder = self._load_vae_decoder(vae_dir)

        self.pipeline = BatchCausalInferencePipeline(
            config=self.config,
            predictor=predictor,
            vae_decoder=vae_decoder,
            device="cuda"
        ).to(device=self.device, dtype=self.weight_dtype)

        self.pipeline.vae_decoder.to(torch.float16)

        self.vae_encoder = create_wan_encoder(
            vae_dir,
            self.device,
            self.weight_dtype
        )

    def _load_vae_decoder(self, vae_dir, compile_mode: VAECompileMode = VAECompileMode.AUTO):
        vae_decoder = VaeDecoderWrapper()

        compiled_model_path = os.path.join(vae_dir, "compiled_vae_decoder.pt")
        vae_state_dict = torch.load(os.path.join(vae_dir, "Wan2.1_VAE.pth"), map_location="cpu")
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

        return vae_decoder
    
    def _resizecrop(self, image:PIL.Image.Image,, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image

    def generate(self, image:PIL.Image.Image, num_frames=150, seed=0):
        image = self._resizecrop(image, 352, 640)
        image_tensor = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)

        padding_video = torch.zeros_like(image_tensor).repeat(1, 1, 4 * (num_frames - 1), 1, 1)
        img_cond = torch.concat([image_tensor, padding_video], dim=2)
        tiler_kwargs={"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae_encoder.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)

        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)

        visual_context = self.vae_encoder.clip.encode_video(image)

        sampled_noise = torch.randn(
            [1, 16, num_frames, 44, 80], device=self.device, dtype=self.weight_dtype
        )

        num_video_frames = (num_frames - 1) * 4 + 1

        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
        }

        cond_data = Bench_actions_universal(num_video_frames)
        mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        conditional_dict['mouse_cond'] = mouse_condition
        keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        conditional_dict['keyboard_cond'] = keyboard_condition

        videos = self.pipeline.inference(
            noise=sampled_noise,
            conditional_dict=conditional_dict,
            return_latents=False,
        )

        assert isinstance(videos, list)
        videos_tensor = torch.cat(videos, dim=1)

        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)

        return video
