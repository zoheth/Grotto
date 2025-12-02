import flashinfer
import torch
import torch.nn as nn
from einops import rearrange


def _flash_attention(q, k, v, causal=False, window_size=(-1, -1)):
    """
    FlashInfer attention from first principles.

    Args:
        q: [B, L_q, H, D]
        k: [B, L_k, H, D]
        v: [B, L_k, H, D]
        causal: bool
        window_size: (left, right) - sliding window

    Returns:
        [B, L_q, H, D]
    """
    B, L_q, H, D = q.shape

    # FlashInfer processes each sample independently
    outputs = []
    for i in range(B):
        out_i = flashinfer.single_prefill_with_kv_cache(
            q=q[i],  # [L_q, H, D]
            k=k[i],  # [L_k, H, D]
            v=v[i],  # [L_k, H, D]
            causal=causal,
            kv_layout="NHD",
            pos_encoding_mode="NONE",  # RoPE applied externally
            window_left=window_size[0] if window_size != (-1, -1) else -1,
        )
        outputs.append(out_i)

    return torch.stack(outputs, dim=0)


class I2VCrossAttention(nn.Module):
    """
    Cross-attention for image-to-video with internal KV caching.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        qk_norm=True,
        eps=1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Linear projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # Q/K normalization
        self.norm_q = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self._kv_cache = None

    def reset_cache(self):
        """Clear cache for new sequence."""
        self._kv_cache = None

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        """
        Args:
            x: [B, L_q, C] - Video latents
            context: [B, L_kv, C] - Image context

        Returns:
            [B, L_q, C]
        """
        H = self.num_heads

        q = self.norm_q(self.q(x))
        q = rearrange(q, "b l (h d) -> b l h d", h=H)

        # Key/Value (cached after first forward)
        if self._kv_cache is None:
            k = self.norm_k(self.k(context))
            v = self.v(context)

            k = rearrange(k, "b l (h d) -> b l h d", h=H)
            v = rearrange(v, "b l (h d) -> b l h d", h=H)

            self._kv_cache = (k, v)
        else:
            k, v = self._kv_cache

        out = _flash_attention(q, k, v)

        out = rearrange(out, "b l h d -> b l (h d)")

        return self.o(out.flatten(2))
