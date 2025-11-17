"""
Configuration dataclasses for the inference pipeline.

This module encapsulates all magic numbers and configuration parameters
into type-safe, well-documented dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration for the WAN diffusion model architecture.

    These parameters define the model structure and should match
    the architecture used during training.
    """

    num_transformer_blocks: int = 30
    """Number of transformer blocks in the model"""

    frame_seq_length: int = 880
    """Sequence length per frame (H*W / patch_size^2)"""

    num_attention_heads: int = 12
    """Number of attention heads in transformer blocks"""

    head_dim: int = 128
    """Dimension of each attention head"""

    num_action_attention_heads: int = 16
    """Number of attention heads for action conditioning"""

    action_head_dim: int = 64
    """Dimension of each action attention head"""

    cross_attention_seq_length: int = 257
    """Sequence length for cross-attention (CLIP features)"""

    @property
    def total_attention_dim(self) -> int:
        """Total dimension of attention (num_heads * head_dim)"""
        return self.num_attention_heads * self.head_dim


@dataclass
class CacheConfig:
    """
    Configuration for KV cache management.

    The KV cache stores key-value pairs from previous frames
    to enable efficient autoregressive generation.
    """

    local_attn_size: int
    """Size of the local attention window (in frames)"""

    use_global_cache: bool = True
    """Whether to use global attention cache"""

    use_mouse_cache: bool = True
    """Whether to cache mouse action conditioning"""

    use_keyboard_cache: bool = True
    """Whether to cache keyboard action conditioning"""

    def get_visual_cache_size(self, frame_seq_length: int) -> int:
        """
        Calculate the total visual KV cache size.

        Args:
            frame_seq_length: Sequence length per frame

        Returns:
            Total cache size in tokens
        """
        return self.local_attn_size * frame_seq_length

    def get_action_cache_size(self) -> int:
        """
        Calculate the action conditioning cache size.

        Returns:
            Action cache size in tokens
        """
        return self.local_attn_size


@dataclass
class VAEConfig:
    """
    Configuration for the VAE encoder/decoder.

    The VAE compresses video frames into a latent space.
    """

    latent_channels: int = 16
    """Number of channels in the latent representation"""

    temporal_compression: int = 4
    """Temporal compression factor (frames)"""

    spatial_compression: int = 8
    """Spatial compression factor (pixels)"""

    def get_latent_frame_count(self, video_frames: int) -> int:
        """
        Convert video frame count to latent frame count.

        Args:
            video_frames: Number of video frames

        Returns:
            Number of latent frames (accounting for compression and padding)
        """
        # Formula: 1 (initial frame) + (video_frames - 1) // temporal_compression
        if video_frames == 0:
            return 0
        return 1 + (video_frames - 1) // self.temporal_compression

    def get_video_frame_count(self, latent_frames: int) -> int:
        """
        Convert latent frame count to video frame count.

        Args:
            latent_frames: Number of latent frames

        Returns:
            Number of video frames
        """
        if latent_frames == 0:
            return 0
        return 1 + (latent_frames - 1) * self.temporal_compression

    def get_action_condition_length(self, latent_frames: int) -> int:
        """
        Get the action condition sequence length for a given number of latent frames.

        The action condition is denser than the latent representation.

        Args:
            latent_frames: Number of latent frames

        Returns:
            Action condition sequence length
        """
        if latent_frames == 0:
            return 0
        return 1 + self.temporal_compression * (latent_frames - 1)


@dataclass
class InferenceConfig:
    """
    Configuration for the inference process.

    This controls how the diffusion process runs.
    """

    denoising_steps: list[int] = field(default_factory=lambda: [1000, 750, 500, 250])
    """List of timesteps for denoising (reversed diffusion)"""

    warp_denoising_step: bool = True
    """Whether to warp timesteps using the scheduler's timestep mapping"""

    context_noise: int = 0
    """Noise level when caching context (0 = clean)"""

    num_frame_per_block: int = 1
    """Number of frames to generate per block"""

    def validate(self, vae_config: VAEConfig) -> None:
        """
        Validate inference config against VAE config.

        Args:
            vae_config: VAE configuration to validate against

        Raises:
            ValueError: If configuration is invalid
        """
        if self.num_frame_per_block < 1:
            raise ValueError("num_frame_per_block must be >= 1")

        if len(self.denoising_steps) == 0:
            raise ValueError("denoising_steps cannot be empty")

        if not all(0 <= step <= 1000 for step in self.denoising_steps):
            raise ValueError("All denoising steps must be in range [0, 1000]")


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.

    This combines all sub-configurations into a single object.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model architecture configuration"""

    cache: CacheConfig = field(default_factory=lambda: CacheConfig(local_attn_size=15))
    """KV cache configuration"""

    vae: VAEConfig = field(default_factory=VAEConfig)
    """VAE configuration"""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    """Inference process configuration"""

    mode: str = 'universal'
    """Game mode: 'universal', 'gta_drive', or 'templerun'"""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mode not in ['universal', 'gta_drive', 'templerun']:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.inference.validate(self.vae)

    @classmethod
    def from_legacy_args(cls, args) -> 'PipelineConfig':
        """
        Create PipelineConfig from legacy args object.

        This is a compatibility layer for migrating from the old code.

        Args:
            args: Legacy args object (e.g., from OmegaConf)

        Returns:
            PipelineConfig instance
        """
        # Extract model config
        model_kwargs = getattr(args, "model_kwargs", {})
        model_config_path = model_kwargs.get("model_config", "")

        # For now, use defaults - in a real migration, you'd parse model_config
        model_config = ModelConfig()

        # Extract cache config
        local_attn_size = getattr(args, "local_attn_size", 15)
        cache_config = CacheConfig(local_attn_size=local_attn_size)

        # Extract inference config
        denoising_steps = getattr(args, "denoising_step_list", [1000, 750, 500, 250])
        warp_denoising = getattr(args, "warp_denoising_step", True)
        context_noise = getattr(args, "context_noise", 0)
        num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        inference_config = InferenceConfig(
            denoising_steps=denoising_steps,
            warp_denoising_step=warp_denoising,
            context_noise=context_noise,
            num_frame_per_block=num_frame_per_block
        )

        # Extract VAE config (using defaults for now)
        vae_config = VAEConfig()

        # Extract mode
        mode = getattr(args, "mode", "universal")

        return cls(
            model=model_config,
            cache=cache_config,
            vae=vae_config,
            inference=inference_config,
            mode=mode
        )
