from typing import Optional

import torch

class FlowMatchScheduler:
    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas

        self.sigmas = torch.tensor([])
        self.timesteps = torch.tensor([])
        self.linear_timesteps_weights = torch.tensor([])

    def _calculate_sigmas(self, num_steps: int, denoising_strength: float = 1.0) -> torch.Tensor:
        sigma_start = self.sigma_min + \
            (self.sigma_max - self.sigma_min) * denoising_strength
        
        if self.extra_one_step:
            sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_steps + 1)[:-1]
        else:
            sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_steps)
            
        if self.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
            
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        
        if self.reverse_sigmas:
            sigmas = 1 - sigmas
            
        return sigmas

    def set_inference_timesteps(self, num_inference_steps: int = 100, 
                                denoising_strength: float = 1.0, 
                                device: Optional[torch.device] = None):
        self.sigmas = self._calculate_sigmas(num_inference_steps, denoising_strength)
        self.timesteps = self.sigmas * self.num_train_timesteps
        
        if device:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)
   
        self.linear_timesteps_weights = torch.tensor([])

    def set_training_timesteps(self, num_steps: int = 1000, device: Optional[torch.device]  = None):

        self.sigmas = self._calculate_sigmas(num_steps, denoising_strength=1.0)
        self.timesteps = self.sigmas * self.num_train_timesteps

        x = self.timesteps
        y = torch.exp(-2 * ((x - num_steps / 2) / num_steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (num_steps / y_shifted.sum())
        
        self.linear_timesteps_weights = bsmntw_weighing
        
        if device:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)
            self.linear_timesteps_weights = self.linear_timesteps_weights.to(device)