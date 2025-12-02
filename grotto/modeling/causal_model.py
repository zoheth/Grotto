import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch import nn

from grotto.modeling.causal_self_attention import CausalSelfAttention
from grotto.modeling.cross_attention import I2VCrossAttention
from grotto.modeling.modular_action import ActionConfig, ActionContext, ActionModule
from grotto.profiling import record_module

if TYPE_CHECKING:
    from .kv_cache import DualPlaneKVCache
    from .ring_buffer_cache import RingBufferActionCache


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        half = dim // 2

        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [Batch] or [Batch, Frames]
        args = t.float().unsqueeze(-1) * self.freqs[None, :]  # type: ignore
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class CausalWanAttentionBlock(nn.Module):
    """
    Transformer block with self-attention, cross-attention, optional action injection, and FFN.

    Architecture:
        x → [AdaLN + Self-Attention + Gate] → [Cross-Attention] → [Action?] → [AdaLN + FFN + Gate] → out
    """

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        num_frame_per_block,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        action_config=None,
        block_idx=0,
        eps=1e-6,
        workspace_buffer: Optional[torch.Tensor] = None,
    ):
        if action_config is None:
            action_config = {}
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        if len(action_config) != 0 and block_idx in action_config["blocks"]:
            config_dict = {**action_config, "local_attn_size": local_attn_size}
            self.action_model = ActionModule(ActionConfig.from_dict(config_dict))
        else:
            self.action_model = None

        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.norm3 = (
            nn.LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )

        self.self_attn = CausalSelfAttention(
            dim,
            num_heads,
            num_frame_per_block,
            local_attn_size,
            sink_size,
            qk_norm,
            eps,
            workspace_buffer=workspace_buffer,
        )

        self.cross_attn = I2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )

        # AdaLN modulation parameters [1, 6, C] for (shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        ada_params: torch.Tensor,
        grid_sizes: Tuple[int, int, int],
        freqs: torch.Tensor,
        context: torch.Tensor,
        kv_cache: "DualPlaneKVCache",
        current_start: int = 0,
        cache_start: Optional[int] = None,
        action_context: Optional[ActionContext] = None,
        cache_mode: str = "read_write",
        incoming_len: Optional[int] = None,
    ) -> torch.Tensor:
        r"""
        Forward pass through the attention block.

        Args:
            x: Hidden states [B, L, C]
            ada_params: AdaLN modulation parameters [B, F, 6, C] where F = num_frames
            grid_sizes: Spatial-temporal grid (num_frames, height, width) as tuple
            freqs: RoPE frequencies [max_len, head_dim/2]
            context: Visual context for cross-attention
            block_mask: Attention mask for self-attention
            kv_cache: DualPlaneKVCache for self-attention
            current_start: Current position in sequence (for cache indexing)
            action_context: Optional ActionContext encapsulating all action-related parameters
            incoming_len: Pre-computed sequence length (avoids shape access)

        Returns:
            Updated hidden states [B, L, C]
        """
        x.shape[1]
        num_frames = ada_params.shape[1]

        # Combine learned modulation with input modulation
        # [1, 6, C] + [B, F, 6, C] → [B, F, 6, C]
        combined_modulation = self.modulation.unsqueeze(1) + ada_params

        # Split into 6 components: [B, F, 1, C] each after chunking
        (shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn) = rearrange(
            combined_modulation, "b f six c -> six b f 1 c"
        )

        # Self-Attention block
        with record_module("Self-Attention"):
            x_mod = self._adaln_modulate(self.norm1(x), shift_msa, scale_msa, f=num_frames)

            # Execute phase: GPU operations (planning is done inside self_attn using pre-computed incoming_len)
            y = self.self_attn(
                x_mod,
                grid_sizes,
                freqs,
                kv_cache,
                current_start,
                cache_mode=cache_mode,
                incoming_len=incoming_len,
            )
            x = self._adaln_gated_residual(x, y, gate_msa, f=num_frames)

        # Cross-Attention and Action blocks
        x = self._apply_condition_attn(x, context, grid_sizes, current_start, action_context)

        # FFN block
        with record_module("FFN"):
            x_mod = self._adaln_modulate(self.norm2(x), shift_ffn, scale_ffn, f=num_frames)
            y = self.ffn(x_mod)
            x = self._adaln_gated_residual(x, y, gate_ffn, f=num_frames)

        return x

    def _adaln_modulate(
        self, x_normed: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, f: int
    ) -> torch.Tensor:
        """
        Input:  x_normed [B, T, C], shift/scale [B, F, 1, C]
        Output: [B, T, C] (Modulated)
        """
        x_view = rearrange(x_normed, "b (f l) c -> b f l c", f=f)
        x_mod = x_view * (1 + scale) + shift
        return rearrange(x_mod, "b f l c -> b (f l) c")

    def _adaln_gated_residual(
        self, x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor, f: int
    ) -> torch.Tensor:
        """
        Input:  x [B, T, C] (Residual), y [B, T, C] (Branch), gate [B, F, 1, C]
        Output: [B, T, C] (x + gate * y)
        """
        y_view = rearrange(y, "b (f l) c -> b f l c", f=f)
        y_gated = rearrange(y_view * gate, "b f l c -> b (f l) c")
        return x + y_gated

    def _apply_condition_attn(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: tuple,  # (F, H, W)
        current_start: int,
        action_context: Optional[ActionContext],
    ) -> torch.Tensor:
        with record_module("CLIP Cross-Attn"):
            x = x + self.cross_attn(self.norm3(x.to(context.dtype)), context)

        if self.action_model is not None:
            if action_context is None or not action_context.has_any_condition:
                raise ValueError(
                    "ActionModule is enabled but no ActionContext provided. "
                    "Either pass action_context or use legacy action_kwargs."
                )

            spatial_tokens_per_frame = int(grid_sizes[1] * grid_sizes[2])
            start_frame = current_start // spatial_tokens_per_frame

            with record_module("Action Module"):
                x = self.action_model(
                    x.to(context.dtype),
                    grid_sizes,
                    rotation=action_context.rotation_cond,
                    translation=action_context.translation_cond,
                    is_causal=True,
                    kv_cache_rotation=action_context.kv_cache_rotation,
                    kv_cache_translation=action_context.kv_cache_translation,
                    start_frame=start_frame,
                    num_frame_per_block=action_context.num_frame_per_block,
                )

        return x


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C] - Input features (L1 = F * S)
            e(Tensor): Shape [B, F, 1, C] - Conditioning / AdaLN parameters
        """
        combined_style = e + self.modulation.unsqueeze(1)

        shift, scale = rearrange(combined_style, "b f two c -> two b f 1 c", two=2)

        x = self.norm(x)

        x = rearrange(x, "b (f s) c -> b f s c", f=e.shape[1])

        x = x * (1 + scale) + shift

        return self.head(x)


def get_rope_freqs_complex(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


class CausalWanModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="i2v",
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
        num_frame_per_block=3,
        qk_norm=True,
        cross_attn_norm=True,
        action_config=None,
        eps=1e-6,
    ):
        if action_config is None:
            action_config = {}
        super().__init__()

        assert model_type == "i2v", "Only 'i2v' model_type is supported in CausalWanModel"
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
        self.num_frame_per_block = num_frame_per_block
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.pos_encoder = SinusoidalEmbedding(self.freq_dim)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    self.num_frame_per_block,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    action_config=action_config,
                    eps=eps,
                    block_idx=idx,
                    workspace_buffer=self.workspace_buffer,
                )
                for idx in range(num_layers)
            ]
        )

        self.head = CausalHead(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        head_dim = dim // num_heads

        split_size = head_dim // 6
        dim_h = split_size * 2
        dim_w = split_size * 2
        dim_t = head_dim - dim_h - dim_w
        max_pos = 1024
        self.freqs = torch.cat(
            [
                get_rope_freqs_complex(max_pos, dim_t),
                get_rope_freqs_complex(max_pos, dim_h),
                get_rope_freqs_complex(max_pos, dim_w),
            ],
            dim=1,
        )

        img_dim = 1280
        self.img_emb = torch.nn.Sequential(
            torch.nn.LayerNorm(img_dim),
            torch.nn.Linear(img_dim, img_dim),
            torch.nn.GELU(),
            torch.nn.Linear(img_dim, dim),
            torch.nn.LayerNorm(dim),
        )

        self.init_weights()

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        visual_context,
        cond_concat,
        kv_cache: List["DualPlaneKVCache"],
        action_context: Optional[ActionContext] = None,
        kv_cache_mouse: Optional[List["RingBufferActionCache"]] = None,
        kv_cache_keyboard: Optional[List["RingBufferActionCache"]] = None,
        current_start: int = 0,
        cache_mode: str = "read_write",
    ):
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = torch.cat([x, cond_concat], dim=1)  # B C' F H W

        x = self.patch_embedding(x)
        grid_sizes = tuple(x.shape[2:])
        x = rearrange(x, "b c f h w -> b (f h w) c")
        assert x.shape[1] <= 15 * 1 * 880

        e_raw = self.pos_encoder(timesteps.flatten())
        e: torch.Tensor = self.time_embedding(e_raw.type_as(x))

        e_proj = self.time_projection(e)  # Output: [Total_Batch, 6 * dim]

        e0 = e_proj.unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=timesteps.shape)

        context = self.img_emb(visual_context)

        # Pre-compute incoming_len once (avoids repeated .shape access and potential CPU-GPU sync)
        incoming_len = x.shape[1]
        current_end = current_start + incoming_len

        # Batch planning: Plan KV cache and attention for ALL blocks upfront
        # This consolidates all CPU operations and potential sync points into one place
        for block_index, block in enumerate(self.blocks):
            block.self_attn.plan_kv_and_attention(
                incoming_len=incoming_len,
                kv_cache=kv_cache[block_index],
                current_start=current_start,
                current_end=current_end,
                grid_sizes=grid_sizes,
                cache_mode=cache_mode,
            )

        # Execute all blocks (pure GPU operations, no more planning/sync inside)
        for block_index, block in enumerate(self.blocks):
            if action_context is not None:
                action_context.kv_cache_mouse = (
                    kv_cache_mouse[block_index] if kv_cache_mouse else None
                )
                action_context.kv_cache_keyboard = (
                    kv_cache_keyboard[block_index] if kv_cache_keyboard else None
                )

            x = block(
                x,
                ada_params=e0,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                action_context=action_context,
                kv_cache=kv_cache[block_index],
                current_start=current_start,
                cache_mode=cache_mode,
                incoming_len=incoming_len,
            )

        x = self.head(x, e.reshape(*timesteps.shape, 1, -1))
        x = self.unpatchify(x, grid_sizes)
        return x

    def unpatchify(self, x, grid_sizes):
        f, h, w = grid_sizes
        pt, ph, pw = self.patch_size

        return rearrange(
            x,
            "b f (h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)",
            f=f,
            h=h,
            w=w,
            pt=pt,
            ph=ph,
            pw=pw,
            c=self.out_dim,
        )

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))

        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        if self.use_action_module:
            for block in self.blocks:
                if block.action_model is not None:
                    block.action_model.init_weights()  # type: ignore
