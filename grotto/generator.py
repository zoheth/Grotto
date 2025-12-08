import logging
import os
from enum import Enum

import numpy as np
import PIL.Image
import torch
from einops import rearrange
from safetensors.torch import load_file
from torchvision.transforms import v2

# from grotto.camera_control import generate_camera_navigation
from grotto.camera_control import generate_left_right_sequence
from grotto.modeling.predictor import WanDiffusionPredictor
from grotto.modeling.vae_wrapper import VaeDecoderWrapper, create_wan_encoder
from grotto.modeling.weight_mapping_config import (
    PREDICTOR_IMG_EMB_MAPPING,
    apply_mapping,
    detect_old_predictor_format,
)
from grotto.pipeline import BatchCausalInferencePipeline, PipelineConfig
from grotto.profiling import record_module
from grotto.types import ConditionalInputs


class VAECompileMode(str, Enum):
    AUTO = "auto"
    FORCE = "force"
    NONE = "none"


class VideoGenerator:
    def __init__(
        self,
        config_path,
        checkpoint_path,
        vae_dir,
        device="cuda",
        vae_compile_mode: VAECompileMode = VAECompileMode.AUTO,
    ):
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16
        self.config: PipelineConfig = PipelineConfig.load(config_path)

        self._init_models(checkpoint_path, vae_dir, vae_compile_mode)

        self.frame_process = v2.Compose(
            [
                v2.Resize(size=(352, 640), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _init_models(
        self, checkpoint_path, vae_dir, vae_compile_mode: VAECompileMode = VAECompileMode.AUTO
    ):
        predictor = WanDiffusionPredictor(
            num_frame_per_block=self.config.inference.num_frame_per_block,
            model_config_path=self.config.model_config_path,
            timestep_shift=self.config.inference.timestep_shift,
        )

        state_dict = load_file(checkpoint_path)

        # Apply weight mapping if needed for old checkpoint format
        if detect_old_predictor_format(state_dict):
            logging.info("Detected old checkpoint format, applying weight mapping...")
            state_dict = apply_mapping(state_dict, PREDICTOR_IMG_EMB_MAPPING)

        # Use strict=False to allow missing buffers that are auto-initialized (e.g., freqs)
        predictor.load_state_dict(state_dict, strict=False)

        vae_decoder = self._load_vae_decoder(vae_dir, vae_compile_mode)

        self.pipeline = BatchCausalInferencePipeline(
            config=self.config, predictor=predictor, vae_decoder=vae_decoder, device="cuda"
        ).to(device=self.device, dtype=self.weight_dtype)

        self.pipeline.vae_decoder.to(torch.float16)

        self.vae_encoder = create_wan_encoder(vae_dir, self.device, self.weight_dtype)

    def _load_vae_decoder(self, vae_dir, compile_mode: VAECompileMode = VAECompileMode.AUTO):
        vae_decoder = VaeDecoderWrapper()

        compiled_model_path = os.path.join(vae_dir, "compiled_vae_decoder.pt")
        vae_state_dict = torch.load(
            os.path.join(vae_dir, "Wan2.1_VAE.pth"), map_location="cpu", weights_only=False
        )
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if "decoder." in key or "conv2" in key:
                decoder_state_dict[key] = value

        vae_decoder.load_state_dict(decoder_state_dict)
        vae_decoder.to(device=self.device, dtype=torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()

        if compile_mode == VAECompileMode.NONE:
            logging.info("VAE decoder compilation skipped as per user request.")
        elif compile_mode == VAECompileMode.FORCE:
            logging.info("Forcing VAE decoder compilation...")
            vae_decoder.compile(mode="max-autotune-no-cudagraphs")
            torch.save(vae_decoder, compiled_model_path)
            logging.info(f"Compiled VAE decoder saved to {compiled_model_path}")
        elif compile_mode == VAECompileMode.AUTO:
            if os.path.exists(compiled_model_path):
                logging.info(f"Loading compiled VAE decoder from {compiled_model_path}...")
                vae_decoder = torch.load(
                    compiled_model_path, map_location=self.device, weights_only=False
                )
            else:
                logging.info("Compiling VAE decoder...")
                vae_decoder.compile(mode="max-autotune-no-cudagraphs")
                torch.save(vae_decoder, compiled_model_path)
                logging.info(f"Compiled VAE decoder saved to {compiled_model_path}")

        return vae_decoder

    def _resizecrop(self, image: PIL.Image.Image, target_h: int, target_w: int):
        w, h = image.size
        target_ratio = target_w / target_h

        if w / h > target_ratio:
            new_w, new_h = int(h * target_ratio), h
        else:
            new_w, new_h = w, int(w / target_ratio)

        left, top = (w - new_w) / 2, (h - new_h) / 2
        return image.crop((left, top, left + new_w, top + new_h))

    @torch.no_grad()
    def generate(self, image: PIL.Image.Image, num_frames=150, seed=0):
        # Image preprocessing
        with record_module("Image Preprocessing"):
            image = self._resizecrop(image, 352, 640)
            image_tensor = self.frame_process(image)[None, :, None, :, :].to(
                dtype=self.weight_dtype, device=self.device
            )

        # VAE encoding
        with record_module("VAE Encoder"):
            padding_video = torch.zeros_like(image_tensor).repeat(1, 1, 4 * (num_frames - 1), 1, 1)
            img_cond = torch.concat([image_tensor, padding_video], dim=2)
            vae_config = self.config.vae
            img_cond = self.vae_encoder.encode(
                img_cond,
                device=self.device,
                tiled=vae_config.use_tiling,
                tile_size=list(vae_config.tile_size),
                tile_stride=list(vae_config.tile_stride),
            ).to(self.device)

        # Prepare conditions
        with record_module("Condition Preparation"):
            mask_cond = torch.ones_like(img_cond)
            mask_cond[:, :, 1:] = 0
            cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)

        # CLIP encoding
        with record_module("CLIP Encoder"):
            visual_context = self.vae_encoder.clip.encode_video(image_tensor)

        # Noise and camera control
        with record_module("Setup (Noise & Camera)"):
            sampled_noise = torch.randn(
                [1, 16, num_frames, 44, 80], device=self.device, dtype=self.weight_dtype
            )
            num_video_frames = (num_frames - 1) * 4 + 1
            # camera_control = generate_camera_navigation(num_video_frames).unsqueeze_batch()
            camera_control = generate_left_right_sequence(num_video_frames).unsqueeze_batch()

            conditional_inputs = ConditionalInputs(
                cond_concat=cond_concat,
                visual_context=visual_context,
                rotation_cond=camera_control.rotation,
                translation_cond=camera_control.translation,
            ).to(device=self.device, dtype=self.weight_dtype)

        # Diffusion inference
        with record_module("Diffusion Pipeline"):
            videos = self.pipeline.inference(
                noise=sampled_noise,
                conditional_inputs=conditional_inputs,
                return_latents=False,
            )

        # Post-processing
        with record_module("Post-processing"):
            assert isinstance(videos, list)
            videos_tensor = torch.cat(videos, dim=1)

            videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
            videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
            video = np.ascontiguousarray(videos)

        return video
