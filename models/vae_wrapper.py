from typing import List, Tuple, Optional
import torch
from torch import nn

from .vae import VaeDecoder3d, CacheState, CausalConv3d
from .weight_mapping_config import apply_mapping, detect_old_vae_format, VAE_DECODER_MAPPING

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
