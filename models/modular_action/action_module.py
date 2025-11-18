"""
Modular Action Module - 完整实现
兼容原始 ActionModule 的接口，但使用模块化的内部实现
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from wan.modules.posemb_layers import get_nd_rotary_pos_embed
from .action_config import ActionConfig
from .injectors import MouseInjector, KeyboardInjector


class ActionModule(nn.Module):
    """
    模块化的 Action Module，支持鼠标和键盘条件注入

    与原始 ActionModule 兼容，但使用更清晰的内部架构：
    - MouseInjector: 处理鼠标条件
    - KeyboardInjector: 处理键盘条件
    """

    def __init__(
        self,
        # 基础参数（兼容原始接口）
        mouse_dim_in: int = 2,
        keyboard_dim_in: int = 6,
        hidden_size: int = 128,
        img_hidden_size: int = 1536,
        keyboard_hidden_dim: int = 1024,
        mouse_hidden_dim: int = 1024,
        vae_time_compression_ratio: int = 4,
        windows_size: int = 3,
        heads_num: int = 16,
        patch_size: list = None,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        rope_dim_list: list = None,
        rope_theta: int = 256,
        mouse_qk_dim_list: list = None,
        enable_mouse: bool = True,
        enable_keyboard: bool = True,
        local_attn_size: int = 6,
        blocks: list = None,
        # 新增：可选的 ActionConfig（优先级更高）
        action_config: Optional[ActionConfig] = None,
    ):
        """
        初始化 ActionModule

        Args:
            可以通过两种方式配置：
            1. 使用独立参数（兼容原始接口）
            2. 使用 ActionConfig 对象（推荐）
        """
        super().__init__()

        # 如果提供了 action_config，优先使用它
        if action_config is not None:
            config = action_config
        else:
            # 否则从独立参数构建配置
            if patch_size is None:
                patch_size = [1, 2, 2]
            if rope_dim_list is None:
                rope_dim_list = [8, 28, 28]
            if mouse_qk_dim_list is None:
                mouse_qk_dim_list = [8, 28, 28]
            if blocks is None:
                blocks = []

            config = ActionConfig(
                blocks=blocks,
                enable_keyboard=enable_keyboard,
                enable_mouse=enable_mouse,
                heads_num=heads_num,
                hidden_size=hidden_size,
                img_hidden_size=img_hidden_size,
                keyboard_dim_in=keyboard_dim_in,
                keyboard_hidden_dim=keyboard_hidden_dim,
                mouse_dim_in=mouse_dim_in,
                mouse_hidden_dim=mouse_hidden_dim,
                mouse_qk_dim_list=mouse_qk_dim_list,
                patch_size=patch_size,
                qk_norm=qk_norm,
                qkv_bias=qkv_bias,
                rope_dim_list=rope_dim_list,
                rope_theta=rope_theta,
                vae_time_compression_ratio=vae_time_compression_ratio,
                windows_size=windows_size,
                local_attn_size=local_attn_size,
            )

        self.config = config
        self.enable_mouse = config.enable_mouse
        self.enable_keyboard = config.enable_keyboard
        self.vae_time_compression_ratio = config.vae_time_compression_ratio
        self.windows_size = config.windows_size
        self.patch_size = config.patch_size
        self.rope_dim_list = config.rope_dim_list
        self.rope_theta = config.rope_theta
        self.heads_num = config.heads_num
        self.local_attn_size = config.local_attn_size

        # 初始化 injectors
        if self.enable_mouse:
            self.mouse_injector = MouseInjector(config)
        else:
            self.mouse_injector = None

        if self.enable_keyboard:
            self.keyboard_injector = KeyboardInjector(config)
        else:
            self.keyboard_injector = None

        # 预计算 RoPE embeddings（与原始实现兼容）
        self.freqs_cos, self.freqs_sin = self.get_rotary_pos_embed(
            7500, self.patch_size[1], self.patch_size[2], 64,
            self.rope_dim_list, start_offset=0
        )

    def get_rotary_pos_embed(
        self,
        video_length: int,
        height: int,
        width: int,
        head_dim: int,
        rope_dim_list: list = None,
        start_offset: int = 0
    ):
        """
        计算 RoPE embeddings（与原始 ActionModule 完全一致）

        Args:
            video_length: 视频长度
            height: 高度（patched）
            width: 宽度（patched）
            head_dim: attention head 维度
            rope_dim_list: RoPE 维度分配
            start_offset: 起始偏移

        Returns:
            (freqs_cos, freqs_sin) tuple
        """
        target_ndim = 3
        ndim = 5 - 2

        latents_size = [video_length + start_offset, height, width]

        if isinstance(self.patch_size, int):
            assert all(s % self.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.patch_size for s in latents_size]
        elif isinstance(self.patch_size, list):
            assert all(
                s % self.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // self.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis

        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )

        return (
            freqs_cos[-video_length * rope_sizes[1] * rope_sizes[2] // self.patch_size[0]:],
            freqs_sin[-video_length * rope_sizes[1] * rope_sizes[2] // self.patch_size[0]:]
        )

    def forward(
        self,
        x: torch.Tensor,
        tt: int,
        th: int,
        tw: int,
        mouse_condition: Optional[torch.Tensor] = None,
        keyboard_condition: Optional[torch.Tensor] = None,
        block_mask_mouse: Optional[Any] = None,
        block_mask_keyboard: Optional[Any] = None,
        is_causal: bool = False,
        kv_cache_mouse: Optional[Dict[str, torch.Tensor]] = None,
        kv_cache_keyboard: Optional[Dict[str, torch.Tensor]] = None,
        start_frame: int = 0,
        use_rope_keyboard: bool = True,
        num_frame_per_block: int = 3,
    ) -> torch.Tensor:
        """
        前向传播（与原始 ActionModule 接口完全兼容）

        Args:
            x: [B, T*H*W, C] - 隐藏状态
            tt: 时间维度（features）
            th: 高度（patched）
            tw: 宽度（patched）
            mouse_condition: [B, N_frames, C_mouse] - 鼠标条件
            keyboard_condition: [B, N_frames, C_keyboard] - 键盘条件
            block_mask_mouse: 鼠标 attention mask（暂未使用）
            block_mask_keyboard: 键盘 attention mask（暂未使用）
            is_causal: 是否因果模式
            kv_cache_mouse: 鼠标 KV cache
            kv_cache_keyboard: 键盘 KV cache
            start_frame: 起始帧索引
            use_rope_keyboard: 是否对键盘使用 RoPE（必须为 True）
            num_frame_per_block: 每个 block 的帧数

        Returns:
            [B, T*H*W, C] - 处理后的隐藏状态
        """
        assert use_rope_keyboard == True, "use_rope_keyboard must be True"

        B = x.shape[0]
        assert tt * th * tw == x.shape[1], f"Sequence length mismatch: {tt}*{th}*{tw}={tt*th*tw} != {x.shape[1]}"

        hidden_states = x
        freqs_cis = (self.freqs_cos, self.freqs_sin)

        # Mouse injection
        if self.enable_mouse and mouse_condition is not None:
            hidden_states = self.mouse_injector(
                hidden_states,
                condition=mouse_condition,
                freqs_cis=freqs_cis,
                spatial_shape=(th, tw),
                temporal_shape=tt,
                is_causal=is_causal,
                kv_cache=kv_cache_mouse,
                start_frame=start_frame,
                num_frame_per_block=num_frame_per_block,
            )

        # Keyboard injection
        if self.enable_keyboard and keyboard_condition is not None:
            hidden_states = self.keyboard_injector(
                hidden_states,
                condition=keyboard_condition,
                freqs_cis=freqs_cis,
                spatial_shape=(th, tw),
                temporal_shape=tt,
                is_causal=is_causal,
                kv_cache=kv_cache_keyboard,
                start_frame=start_frame,
                num_frame_per_block=num_frame_per_block,
            )

        return hidden_states

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        重写 PyTorch 的权重加载钩子，自动处理从原始 ActionModule 到模块化版本的键名转换

        这是 PyTorch 推荐的方式，会在递归加载子模块时自动调用
        """
        # 定义键名映射规则
        key_mappings = {
            # Mouse 相关映射
            'mouse_mlp.0.weight': 'mouse_injector.mouse_mlp.0.weight',
            'mouse_mlp.0.bias': 'mouse_injector.mouse_mlp.0.bias',
            'mouse_mlp.2.weight': 'mouse_injector.mouse_mlp.2.weight',
            'mouse_mlp.2.bias': 'mouse_injector.mouse_mlp.2.bias',
            'mouse_mlp.3.weight': 'mouse_injector.mouse_mlp.3.weight',
            'mouse_mlp.3.bias': 'mouse_injector.mouse_mlp.3.bias',
            't_qkv.weight': 'mouse_injector.t_qkv.weight',
            't_qkv.bias': 'mouse_injector.t_qkv.bias',
            'img_attn_q_norm.weight': 'mouse_injector.q_norm.weight',
            'img_attn_k_norm.weight': 'mouse_injector.k_norm.weight',
            'proj_mouse.weight': 'mouse_injector.proj_mouse.weight',
            'proj_mouse.bias': 'mouse_injector.proj_mouse.bias',

            # Keyboard 相关映射
            'keyboard_embed.0.weight': 'keyboard_injector.preprocessor.keyboard_embed.0.weight',
            'keyboard_embed.0.bias': 'keyboard_injector.preprocessor.keyboard_embed.0.bias',
            'keyboard_embed.2.weight': 'keyboard_injector.preprocessor.keyboard_embed.2.weight',
            'keyboard_embed.2.bias': 'keyboard_injector.preprocessor.keyboard_embed.2.bias',
            'mouse_attn_q.weight': 'keyboard_injector.mouse_attn_q.weight',
            'mouse_attn_q.bias': 'keyboard_injector.mouse_attn_q.bias',
            'keyboard_attn_kv.weight': 'keyboard_injector.keyboard_attn_kv.weight',
            'keyboard_attn_kv.bias': 'keyboard_injector.keyboard_attn_kv.bias',
            'key_attn_q_norm.weight': 'keyboard_injector.q_norm.weight',
            'key_attn_k_norm.weight': 'keyboard_injector.k_norm.weight',
            'proj_keyboard.weight': 'keyboard_injector.proj_keyboard.weight',
            'proj_keyboard.bias': 'keyboard_injector.proj_keyboard.bias',
        }

        # 检测是否需要重映射（如果存在旧格式的键）
        needs_remapping = any(
            prefix + old_key in state_dict
            for old_key in key_mappings.keys()
        )

        if needs_remapping:
            # 创建新的 state_dict，进行键名重映射
            new_state_dict = {}
            remapped_count = 0

            for key, value in list(state_dict.items()):
                if key.startswith(prefix):
                    # 移除前缀得到相对键名
                    relative_key = key[len(prefix):]

                    # 检查是否需要映射
                    if relative_key in key_mappings:
                        new_relative_key = key_mappings[relative_key]
                        new_key = prefix + new_relative_key
                        new_state_dict[new_key] = value
                        remapped_count += 1
                        # 从原 state_dict 删除旧键，避免出现在 unexpected_keys 中
                        del state_dict[key]
                    else:
                        # 不需要映射的键保持不变
                        new_state_dict[key] = value
                else:
                    # 不属于当前 prefix 的键保持不变
                    new_state_dict[key] = value

            # 更新 state_dict
            state_dict.update(new_state_dict)

            if remapped_count > 0:
                print(f"[INFO] Remapped {remapped_count} keys from original ActionModule format to modular format")

        # 调用父类的加载逻辑
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __repr__(self) -> str:
        """美化打印"""
        return (
            f"ActionModule(\n"
            f"  enable_mouse={self.enable_mouse},\n"
            f"  enable_keyboard={self.enable_keyboard},\n"
            f"  img_hidden_size={self.config.img_hidden_size},\n"
            f"  heads_num={self.heads_num},\n"
            f"  local_attn_size={self.local_attn_size},\n"
            f"  vae_time_compression_ratio={self.vae_time_compression_ratio},\n"
            f"  windows_size={self.windows_size},\n"
            f")"
        )
