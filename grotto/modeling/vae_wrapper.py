import os
from typing import List, Tuple, Optional
import torch
from torch import nn

from grotto.modeling.vae import VaeDecoder3d, CacheState, CausalConv3d
from grotto.modeling.weight_mapping_config import apply_mapping, detect_old_vae_format, VAE_DECODER_MAPPING
from grotto.modeling.wanx_vae_src import WanVAE, CLIPModel

class VaeDecoderWrapper(nn.Module):
    """Wrapper for new VaeDecoder3d to match the interface expected by inference pipeline"""

    def __init__(self):
        super().__init__()
        self.decoder = VaeDecoder3d(
            dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temporal_upsample=[True, True, False],
            dropout=0.0
        )

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.z_dim = 16

        self.conv2 = CausalConv3d(self.z_dim, self.z_dim, 1)
        self.cache_size = 50

    def forward(
            self,
            z: torch.Tensor,
            *feat_cache: List
    ):
        """
        Args:
            z: Latent tensor [B, T, C, H, W]
            feat_cache: Optional cache states from previous forward pass

        Returns:
            Tuple of (decoded_video [B, T, 3, H', W'], cache_states)
        """
        # Permute to [B, C, T, H, W] and apply normalization
        z = z.permute(0, 2, 1, 3, 4)
        device, dtype = z.device, z.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]

        iter_ = z.shape[2]
        x = self.conv2(z, None)

        is_valid_cache = (len(feat_cache) > 0 and
                         feat_cache[0] is not None and
                         isinstance(feat_cache[0], CacheState))

        if is_valid_cache:
            cache_states = list(feat_cache)
        else:
            cache_states = [CacheState(size=self.cache_size) for _ in range(4)]
            for layer in self.decoder.upsamples:
                if hasattr(layer, 'stream_state'):
                    layer.stream_state.fill_(0) # type: ignore

        out = None
        for i in range(iter_):
            for cache_state in cache_states:
                cache_state.reset_index() # type: ignore

            frame_latent = x[:, :, i:i + 1, :, :]
            frame_output = self.decoder(frame_latent, cache_states=cache_states)

            if out is None:
                out = frame_output
            else:
                out = torch.cat([out, frame_output], 2)

        if out is None:
            batch_size = z.shape[0]
            out = torch.zeros((batch_size, 0, 3, *z.shape[3:]),
                             dtype=z.dtype, device=z.device)
        else:
            out = out.float().clamp_(-1, 1)
            out = out.permute(0, 2, 1, 3, 4)

        return out, cache_states

    def load_state_dict(self, state_dict, strict=True): # type: ignore
        """Load state dict with automatic weight mapping for old format"""
        if detect_old_vae_format(state_dict):
            print("Detected old VAEDecoderWrapper format, applying weight mapping...")
            state_dict = apply_mapping(state_dict, VAE_DECODER_MAPPING)
            strict = False  # Use non-strict mode after mapping

        return super().load_state_dict(state_dict, strict=strict)

class WanVAEEncoder(nn.Module):
    """WAN VAE Encoder with integrated CLIP visual features.

    Combines VAE encoding/decoding with CLIP visual feature extraction
    for video generation tasks.
    """

    def __init__(self, vae, clip):
        super().__init__()
        self.vae = vae
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.clip = clip
        if clip is not None:
            self.clip.requires_grad_(False)
            self.clip.eval()

    def encode(self, x, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """Encode input frames to latent space.

        Args:
            x: Input tensor [B, C, T, H, W]
            device: Target device
            tiled: Whether to use tiled encoding
            tile_size: Size of tiles for tiled encoding
            tile_stride: Stride for tiled encoding

        Returns:
            Encoded latent tensor
        """
        return self.vae.encode(x, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

    def extract_clip_features(self, x):
        """Extract CLIP visual features from input.

        Args:
            x: Input tensor

        Returns:
            CLIP feature tensor
        """
        return self.clip(x)

    def decode(self, latents, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        """Decode latents back to pixel space.

        Args:
            latents: Latent tensor
            device: Target device
            tiled: Whether to use tiled decoding
            tile_size: Size of tiles for tiled decoding
            tile_stride: Stride for tiled decoding

        Returns:
            Decoded video tensor
        """
        return self.vae.decode(latents, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

    def to(self, device, dtype):
        """Move encoder and CLIP to specified device and dtype.

        Args:
            device: Target device
            dtype: Target data type

        Returns:
            self
        """
        self.vae = self.vae.to(device, dtype)

        if self.clip is not None:
            self.clip = self.clip.to(device, dtype)

        return self


def create_wan_encoder(model_path, device, dtype):
    """Create and initialize a WAN VAE encoder with CLIP.

    This encoder combines VAE encoding and CLIP visual feature extraction,
    ready for inference with all models loaded and moved to the target device.

    Args:
        model_path: Path to pretrained model directory
        device: Device to place models on (e.g., 'cuda', 'cpu')
        dtype: Data type for model weights (e.g., torch.float16)

    Returns:
        WanVAEEncoder: Fully initialized encoder ready for inference
    """
    vae = WanVAE(pretrained_path=os.path.join(model_path, "Wan2.1_VAE.pth")).to(dtype)
    clip = CLIPModel(
        checkpoint_path=os.path.join(model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        tokenizer_path=os.path.join(model_path, 'xlm-roberta-large')
    )
    encoder = WanVAEEncoder(vae, clip)
    return encoder.to(device, dtype)
