import copy
from typing import Optional, Dict, List
import torch

from einops import rearrange
from tqdm import tqdm

from .base_pipeline import BaseCausalInferencePipeline
from .config import PipelineConfig
from .cache_manager import CacheManager
from ..models.constant import ZERO_VAE_CACHE

class BatchCausalInferencePipeline(BaseCausalInferencePipeline):

    def __init__(
        self,
        config: PipelineConfig,
        generator,
        vae_decoder,
        device: str = "cuda",
        page_size: int = 16,
    ):
        super().__init__(
            config, generator, vae_decoder, device, page_size=page_size
        )
        

    def _ensure_cache_initialized(self, batch_size: int, dtype: torch.dtype):
        if self.cache_manager is None:
            self.cache_manager = CacheManager(
                model_config=self.config.model,
                cache_config=self.config.cache,
                device= self.device,
                dtype=dtype,
                page_size=self.page_size
            )
            self.cache_manager.initialize_all_caches(batch_size)
        elif not self.cache_manager.is_initialized():
            self.cache_manager.initialize_all_caches(batch_size)
        else:
            # Already initialized, just reset
            self.cache_manager.reset_all_caches()
        
    def _denoise_block(
            self,
            noisy_input: torch.Tensor,
            conditional_dict: Dict[str, torch.Tensor],
            current_start_frame: int,
            batch_size: int
    ) -> torch.Tensor:
        current_num_frames = noisy_input.shape[2]
        visual_cache = self.cache_manager.visual_cache
        current_start = current_start_frame * self.config.model.frame_seq_length

        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones(
                [batch_size, current_num_frames],
                device=self.device,
                dtype=torch.int64
            ) * current_timestep

            flow_pred, _ = self.predictor(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=visual_cache,
                current_start=current_start
            )

            denoised_pred = self.scheduler.convert_flow_to_x0(
                flow_pred=rearrange(flow_pred, 'b c f h w -> (b f) c h w'),
                xt=rearrange(noisy_input, 'b c f h w -> (b f) c h w'),
                timestep=timestep.flatten(0, 1)
            )
            denoised_pred = rearrange(denoised_pred, '(b f) c h w -> b c f h w', b=batch_size)

            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = self.scheduler.add_noise(
                    rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                    torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                    next_timestep * torch.ones(
                        [batch_size * current_num_frames],
                        device=self.device,
                        dtype=torch.long
                    )
                )
                noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=batch_size)

        return denoised_pred # type: ignore


    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        return_latents: bool = False,
        profile: bool = False
    ) -> torch.Tensor | List[torch.Tensor]:
        
        assert noise.shape[1] == self.config.vae.latent_channels
        batch_size, num_channels, num_frames, height, width = noise.shape

        assert num_frames % self.config.inference.num_frame_per_block == 0
        num_blocks = num_frames // self.config.inference.num_frame_per_block

        self._ensure_cache_initialized(batch_size, noise.dtype)

        num_output_frames = num_frames

        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=self.device,
            dtype=noise.dtype
        )

        vae_cache = [None] * len(ZERO_VAE_CACHE)
        videos = []

        current_start_frame = 0

        if profile:
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)

        all_num_frames = [self.config.inference.num_frame_per_block] * num_blocks

        for block_idx, current_num_frames in enumerate(tqdm(all_num_frames)):
            noisy_input = noise[
                :, :,
                current_start_frame : current_start_frame + current_num_frames
            ]

            block_cond, _ = self.condition_processor.slice_block_conditions(
                conditional_dict,
                current_start_frame,
                current_num_frames
            )

            if profile:
                torch.cuda.synchronize()
                diffusion_start.record() # type: ignore

            denoised_pred = self._denoise_block(
                noisy_input,
                block_cond,
                current_start_frame,
                batch_size
            )

            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            