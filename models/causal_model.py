from typing import Optional, Dict, Tuple
from torch import nn
import torch
import math

from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.loaders.peft import PeftAdapterMixin
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from .modular_action.action_module import ActionModule
from .paged_cache import PagedCache

class PrecomputedRoPE3DCache:
    def __init__(
            self,
            freqs: torch.Tensor, # Original freqs [1024, head_dim//2]
            max_frames: int = 1024,
            height: int = 22,  # After patchification (352 / 16)
        width: int = 40,   # After patchification (640 / 16)
        device: Optional[torch.device] = None,
    ):
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.device = device or freqs.device
        self.dtype = freqs.dtype

        head_dim_half = freqs.shape[1]
        c = head_dim_half
        self.c_time = c - 2 * (c // 3)
        self.c_height = c // 3
        self.c_width = c // 3
        self.head_dim_half = head_dim_half

        # Store only the 1D frequency components (memory efficient!)
        # These are small: [1024, c_time/c_height/c_width] each
        self.freqs_time = freqs[:, :self.c_time].to(self.device)       # [1024, c_time]
        self.freqs_height = freqs[:, self.c_time:self.c_time + self.c_height].to(self.device)  # [1024, c_height]
        self.freqs_width = freqs[:, self.c_time + self.c_height:].to(self.device)   # [1024, c_width]

    def get_freqs_for_frame_range(
            self,
            start_frame: int,
            num_frames: int
    ) -> torch.Tensor:
        end_frame = start_frame + num_frames

        if end_frame > self.freqs_time.shape[0]:
            raise ValueError(
                f"Frame range [{start_frame}:{end_frame}] exceeds frequency table size {self.freqs_time.shape[0]}. "
                f"The model's RoPE frequencies don't support this many frames."
            )
        
        F, H, W = num_frames, self.height, self.width

        time_freqs = self.freqs_time[start_frame:end_frame].view(F, 1, 1, -1).expand(F, H, W, -1)  # [F, H, W, c_time]
        height_freqs = self.freqs_height[:H].view(1, H, 1, -1).expand(F, H, W, -1)  # [F, H, W, c_height]
        width_freqs = self.freqs_width[:W].view(1, 1, W, -1).expand(F, H, W, -1)    # [F, H, W, c_width]

        freqs = torch.cat([time_freqs, height_freqs, width_freqs], dim=-1)  # [F, H, W, head_dim//2]

        freqs = freqs.view(F * H * W, 1, self.head_dim_half)  # [F*H*W, 1, head_dim//2]
        
        return freqs
    
def apply_rope_3d_precomputed(
    x: torch.Tensor,  # [B, seq_len, num_heads, head_dim]
    freqs: torch.Tensor,  # [seq_len, 1, head_dim//2]
) -> torch.Tensor:
    B, seq_len, num_heads, head_dim = x.shape

    x_complex = torch.view_as_complex(
        x.reshape(B, seq_len, num_heads, head_dim // 2, 2).to(torch.float64)
    )
    # Apply rotation (broadcast over batch and heads)
    # freqs: [seq_len, 1, head_dim//2]
    # x_complex: [B, seq_len, num_heads, head_dim//2]
    x_rotated = x_complex * freqs.unsqueeze(0) # [B, seq_len, num_heads, head_dim//2]
    x_out = torch.view_as_real(x_rotated).flatten(-2) # [B, seq_len, num_heads, head_dim]
    return x_out.type_as(x)

class FlashInferPlanner:
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        page_size: int,
        workspace_size: int = 128 * 1024 * 1024,  # 128MB
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.workspace_size = workspace_size

        self._workspace_buffer: Optional[torch.Tensor] = None
        self._prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
        self._is_planned = False

    def init(self, device: torch.device) -> None:

        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                self.workspace_size,
                dtype=torch.uint8,
                device=device
            )

        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._workspace_buffer,
                kv_layout="NHD"
            )

    def plan(
        self,
        kv_cache: PagedCache,
        q_len: int,
        device: torch.device,
        q_dtype: torch.dtype,
    ) -> None:
        """
        Execute plan for the current generation step.

        This should be called ONCE per generation step, before any attention layers.
        All layers will share this plan.
        """

        self.init(device)

        # Get FlashInfer metadata from cache
        paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len = kv_cache.get_flashinfer_meta(device)

        # Build qo_indptr for single batch
        qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=device)

        # Plan the paged attention
        self._prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            causal=False,  # Already handled by cache management
            q_data_type=q_dtype,
        )

        self._is_planned = True

    def run(
        self,
        q: torch.Tensor,  # [q_len, num_heads, head_dim]
        kv_cache: PagedCache,
    ) -> torch.Tensor:
        """
        Run paged attention using the pre-computed plan.

        Args:
            q: Query tensor [q_len, num_heads, head_dim]
            kv_cache: Layer-specific PagedCache

        Returns:
            Attention output [q_len, num_heads, head_dim]
        """
        if not self._is_planned:
            raise RuntimeError("FlashInferPlanner.plan() must be called before run()")

        return self._prefill_wrapper.run(
            q,
            (kv_cache.k_cache, kv_cache.v_cache),
        )

    @property
    def is_planned(self) -> bool:
        return self._is_planned

    def reset(self) -> None:
        """Reset plan state. Call at the start of each generation step if needed."""
        self._is_planned = False
        

class CausalSelfAttention(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
        max_frames: int = 1024,
        height: int = 22,
        width: int = 40,
        page_size: int = 16):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Spatial dimensions
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.frame_seq_len = height * width
        self.max_attention_size = (
            15 * self.frame_seq_len if local_attn_size == -1 
            else local_attn_size * self.frame_seq_len
        )

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = nn.RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # RoPE cache (initialized lazily on first forward)
        self.rope_cache: Optional[PrecomputedRoPE3DCache] = None

        self._workspace_buffer: Optional[torch.Tensor] = None
        self._prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None

    def forward(
            self,
            x: torch.Tensor, # [B, seq_len, dim]
            grid_sizes: Tuple[int, int, int],  # (F, H, W) for current frame
            freqs: torch.Tensor,  # [1024, head_dim//2]
            planner: FlashInferPlanner,
            kv_cache: Optional[PagedCache] = None,
            current_start: int = 0,
            cache_start: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass with FlashInfer attention.

        When kv_cache is None: Full attention (training/first frame)
        When kv_cache is provided: Incremental attention with KV cache

        Args:
            x: Input hidden states [B, seq_len, dim]
            seq_lens: Sequence lengths [B]
            grid_sizes: [F, H, W] grid dimensions
            freqs: RoPE frequencies [1024, head_dim//2]
            block_mask: Attention mask (unused with FlashInfer)
            kv_cache: KV cache dictionary
            current_start: Current position in sequence
            cache_start: Start position of cache

        Returns:
            Output hidden states [B, seq_len, dim]
        """
        B, S, H, D = *x.shape[:2], self.num_heads, self.head_dim

        if cache_start is None:
            cache_start = current_start

        q = self.norm_q(self.q(x)).view(B, S, H, D)
        k = self.norm_k(self.k(x)).view(B, S, H, D)
        v = self.v(x).view(B, S, H, D)

        if kv_cache is not None:
            return self._forward_incremental(
                q, k, v, grid_sizes, freqs, planner, kv_cache, current_start, cache_start
            )
        
        else:
            raise NotImplementedError("Full attention not implemented in this snippet.")

    def _init_rope_cache(self, freqs: torch.Tensor):
        if self.rope_cache is None:
            self.rope_cache = PrecomputedRoPE3DCache(
                freqs=freqs,
                max_frames=self.max_frames,
                height=self.height,
                width=self.width,
                device=freqs.device
            )

    def _forward_incremental(
        self,
        q: torch.Tensor,  # [B, seq_len, num_heads, head_dim]
        k: torch.Tensor,
        v: torch.Tensor,
        grid_sizes: Tuple[int, int, int],  # (F, H, W) for current frame
        freqs: torch.Tensor,
        planner: FlashInferPlanner,
        kv_cache: PagedCache,
        current_start: int,
        cache_start: int,
    ) -> torch.Tensor:
        F, H, W = grid_sizes

        self._init_rope_cache(freqs)

        frame_seqlen = H * W
        current_start_frame = current_start // frame_seqlen
        num_frames = F

        assert self.rope_cache is not None
        precomputed_freqs = self.rope_cache.get_freqs_for_frame_range(
            current_start_frame, num_frames
        )

        roped_q = apply_rope_3d_precomputed(q, precomputed_freqs)
        roped_k = apply_rope_3d_precomputed(k, precomputed_freqs)

        roped_k_squeezed = roped_k.squeeze(0)
        v_squeezed = v.squeeze(0)

        current_end = current_start + roped_k_squeezed.shape[0]

        kv_cache.update_or_append(roped_k_squeezed, v_squeezed, current_end)

        kv_cache.evict(self.max_attention_size)

        if not planner.is_planned:
            planner.plan(
                kv_cache=kv_cache,
                    q_len=roped_q.shape[1],
                    device=roped_q.device,
                    q_dtype=roped_q.dtype,
            )

        q_for_flash = roped_q.squeeze(0)  # [q_len, num_heads, head_dim]
        x = planner.run(q_for_flash, kv_cache)
        x = x.unsqueeze(0)  # [1, q_len, num_heads, head_dim]

        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='i2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=36,
                 dim=1536,
                 ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=12,
                 num_layers=30,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 action_config={},
                 eps=1e-6):
        super().__init__()

        assert model_type == 'i2v', "Only 'i2v' model_type is supported in CausalWanModel"
        self.model_type = model_type
        self.use_action_module = len(action_config) > 0
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6)
        )

        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([

        ])