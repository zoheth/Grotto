from typing import Optional, List, Dict
from abc import ABC, abstractmethod
import torch
import copy

from einops import rearrange
from tqdm import tqdm

from grotto.pipeline.config import PipelineConfig
from grotto.pipeline.condition_processor import ConditionProcessor
from grotto.modeling.predictor import WanDiffusionPredictor
from grotto.scheduler import FlowMatchScheduler

class BaseCausalInferencePipeline(torch.nn.Module, ABC):
    """
    Base class for causal inference pipelines.

    This class provides:
    - Model initialization
    - Scheduler setup
    - KV cache management
    - Common inference loop structure

    Subclasses implement specific inference modes (batch vs. streaming).
    """
     
    def __init__(
        self,
        config: PipelineConfig,
        predictor: WanDiffusionPredictor,
        vae_decoder,
        device: str = "cuda",
        page_size: int = 16
    ):
        super().__init__()
        self.config = config
        self.predictor = predictor
        self.vae_decoder = vae_decoder
        self.device = torch.device(device)
        self.page_size = page_size

        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=1000,
            shift=self.config.inference.timestep_shift,
            sigma_min=0.0,
            extra_one_step=True
        )
        self.scheduler.set_timesteps(num_inference_steps=1000)

        self.denoising_step_list = self.scheduler.get_inference_timesteps(
            custom_steps=config.inference.denoising_steps,
            warp=config.inference.warp_denoising_step
        )

        self.cache_manager = None

        self.condition_processor = ConditionProcessor(
            vae_config=config.vae,
            mode=config.mode
        )
        print(f"Initialized {self.__class__.__name__} with {config.inference.num_frame_per_block} frames per block")
