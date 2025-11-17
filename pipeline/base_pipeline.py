from typing import Optional, List, Dict
from abc import ABC, abstractmethod
import torch
import copy

from einops import rearrange
from tqdm import tqdm

from .config import PipelineConfig
from ..models.wan_models import WanDiffusionPredictor
from ..scheduler import FlowMatchScheduler

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
        device: str = "cuda"
    ):
        super().__init__()
        self.config = config
        self.predictor = predictor
        self.vae_decoder = vae_decoder
        self.device = torch.device(device)

        self.scheduler = FlowMatchScheduler