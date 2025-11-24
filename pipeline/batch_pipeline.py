from typing import Optional, Dict, List
import torch

from einops import rearrange
from tqdm import tqdm

from .base_pipeline import BaseCausalInferencePipeline
from .config import PipelineConfig

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

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False
    ) -> torch.Tensor | List[torch.Tensor]:
        
        assert noise.shape[1] == self.config.vae.latent_channels
        batch_size, num_channels, num_frames, height, width = noise.shape

        assert num_frames % self.config.inference.num_frame_per_block == 0
        num_blocks = num_frames // self.config.inference.num_frame_per_block

        