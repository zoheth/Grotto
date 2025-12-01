"""
Configuration dataclasses for the inference pipeline.

This module encapsulates all magic numbers and configuration parameters
into type-safe, well-documented dataclasses.
"""

from dataclasses import dataclass, field

from omegaconf import OmegaConf


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

    use_tiling: bool = True
    tile_size: tuple[int, int] = (44, 80)
    tile_stride: tuple[int, int] = (23, 38)

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

    timestep_shift: float = 5.0
    """Timestep shift applied during inference"""

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

    model_config_path: str

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model architecture configuration"""

    cache: CacheConfig = field(default_factory=lambda: CacheConfig(local_attn_size=15))
    """KV cache configuration"""

    vae: VAEConfig = field(default_factory=VAEConfig)
    """VAE configuration"""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    """Inference process configuration"""

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.inference.validate(self.vae)

    @classmethod
    def load(cls, config_path: str) -> "PipelineConfig":
        config = OmegaConf.load(config_path)

        local_attn_size = config.cache.local_attn_size
        cache_config = CacheConfig(local_attn_size=local_attn_size)

        denoising_steps = config.denoising_steps
        warp_denoising = config.warp_denoising_step
        context_noise = config.context_noise
        num_frame_per_block = config.num_frame_per_block

        inference_config = InferenceConfig(
            denoising_steps=denoising_steps,
            warp_denoising_step=warp_denoising,
            context_noise=context_noise,
            num_frame_per_block=num_frame_per_block,
        )

        # Extract VAE config (using defaults for now)
        vae_config = VAEConfig()

        return cls(
            model_config_path=config.model_config_path,
            cache=cache_config,
            vae=vae_config,
            inference=inference_config,
        )
