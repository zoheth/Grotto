from abc import ABC, abstractmethod
from typing import Optional, List
import torch

class DiffusionScheduler(ABC):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 100
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # These will be set by set_timesteps()
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None

    @abstractmethod
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        denoising_strength: float = 1.0
    ) -> None:
        """
        Set the discrete timesteps for inference.

        Args:
            num_inference_steps: Number of inference steps (uses default if None)
            denoising_strength: Strength of denoising (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to clean samples.

        Args:
            original_samples: Clean samples [B, C, H, W]
            noise: Noise to add [B, C, H, W]
            timestep: Timesteps [B] or [B, T]

        Returns:
            Noisy samples [B, C, H, W]
        """
        pass

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Reverse diffusion: single denoising step.

        Args:
            model_output: Model prediction
            timestep: Current timestep
            sample: Current noisy sample
            **kwargs: Additional scheduler-specific arguments

        Returns:
            Denoised sample for next step
        """
        pass

    def get_inference_timesteps(
        self,
        custom_steps: Optional[List[int]] = None,
        warp: bool = False
    ) -> torch.Tensor:
        """
        Get timesteps for inference, optionally warped or custom.

        This centralizes ALL timestep logic in the scheduler.

        Args:
            custom_steps: Custom timestep values (e.g., [1000, 750, 500, 250])
            warp: Whether to warp custom steps using the scheduler's mapping

        Returns:
            Tensor of timesteps to use for inference
        """
        if custom_steps is None:
            # Use default timesteps from set_timesteps()
            return self.timesteps.clone()

        steps_tensor = torch.tensor(custom_steps, dtype=torch.long)

        if not warp:
            return steps_tensor

        # Warp timesteps using scheduler's mapping
        # This replaces the _prepare_denoising_steps logic in Pipeline
        timesteps_extended = torch.cat([
            self.timesteps.cpu(),
            torch.tensor([0], dtype=torch.float32)
        ])
        warped_steps = timesteps_extended[1000 - steps_tensor]

        return warped_steps

class FlowMatchScheduler(DiffusionScheduler):
    """
    Flow Matching scheduler for continuous-time diffusion.

    This implements the Rectified Flow formulation where:
    - x_t = (1 - sigma_t) * x_0 + sigma_t * noise
    - Velocity prediction: v = noise - x_0
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 100,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False
    ):
        super().__init__(num_train_timesteps, num_inference_steps)

        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas

        # Initialize timesteps
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        denoising_strength: float = 1.0,
        training: bool = False
    ) -> None:
        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps

        # Compute sigma schedule
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength

        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, self.num_inference_steps + 1
            )[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, self.num_inference_steps
            )

        # Apply transformations
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])

        # Time-shift transformation
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)

        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas

        # Map to discrete timesteps
        self.timesteps = self.sigmas * self.num_train_timesteps # type: ignore

        # Compute training weights if needed
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - self.num_inference_steps / 2) / self.num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.training_weights = y_shifted * (self.num_inference_steps / y_shifted.sum())

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Add noise according to flow matching schedule.

        Formula: x_t = (1 - sigma_t) * x_0 + sigma_t * noise

        Args:
            original_samples: Clean samples [B, C, H, W] or [B*T, C, H, W]
            noise: Noise [B, C, H, W] or [B*T, C, H, W]
            timestep: Timesteps [B] or [B*T]

        Returns:
            Noisy samples
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)

        self.sigmas = self.sigmas.to(noise.device) # type: ignore
        self.timesteps = self.timesteps.to(noise.device) # type: ignore

        # Find sigma for each sample
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
            dim=1
        )
        sigma = self.sigmas[timestep_id].view(-1, 1, 1, 1)

        noisy_samples = (1 - sigma) * original_samples + sigma * noise

        return noisy_samples.type_as(noise)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        to_final: bool = False
    ) -> torch.Tensor:
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)

        