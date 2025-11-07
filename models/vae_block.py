from typing import Optional, List

from torch import nn

# Constants
CACHE_FRAMES = 2  # Number of frames to cache for causal convolution
FIRST_CHUNK_MARKER = 'FIRST_CHUNK'  # Marker for first chunk in streaming mode

class VaeDecoder3d(nn.Module):
    def __init__(self,
                 dim: int = 96,
                 z_dim: int = 16,
                 dim_mult: Optional[List[int]] = None,
                 num_res_blocks: int = 2,
                 attn_scales: Optional[List[float]] = None,
                 temporal_upsample: Optional[List[bool]] = None,
                 dropout: float = 0.0
                 ):
        super().__init__()

        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_upsample is None:
            temporal_upsample = [True, True, False]

        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_upsample = temporal_upsample
        self.cache_frames = CACHE_FRAMES

        # Calculate dimensions at each level
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

