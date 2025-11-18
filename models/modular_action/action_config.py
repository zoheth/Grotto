"""
Action Module Configuration
用于管理 ActionModule 的配置参数
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import json


@dataclass
class ActionConfig:
    """
    ActionModule 的配置类

    Attributes:
        blocks: 在哪些 transformer block 中启用 action module
        enable_keyboard: 是否启用键盘条件
        enable_mouse: 是否启用鼠标条件
        heads_num: attention head 的数量
        hidden_size: 隐藏层维度
        img_hidden_size: 图像隐藏层维度
        keyboard_dim_in: 键盘输入的维度
        keyboard_hidden_dim: 键盘 attention 的隐藏维度
        mouse_dim_in: 鼠标输入的维度
        mouse_hidden_dim: 鼠标 attention 的隐藏维度
        mouse_qk_dim_list: 鼠标 RoPE 的维度分配 [T, H, W]
        patch_size: patch 大小 [T, H, W]
        qk_norm: 是否对 query 和 key 做归一化
        qkv_bias: qkv 投影层是否使用 bias
        rope_dim_list: RoPE 的维度分配 [T, H, W]
        rope_theta: RoPE 的 theta 参数
        vae_time_compression_ratio: VAE 在时间维度的压缩比
        windows_size: 滑动窗口大小
        local_attn_size: 局部注意力窗口大小（用于 KV cache）
    """

    # Block 配置
    blocks: List[int] = field(default_factory=lambda: list(range(15)))

    # 模块开关
    enable_keyboard: bool = True
    enable_mouse: bool = True

    # 网络结构参数
    heads_num: int = 16
    hidden_size: int = 128
    img_hidden_size: int = 1024  # 默认值修改为 1024，使 head_dim = 64

    # 键盘相关参数
    keyboard_dim_in: int = 4
    keyboard_hidden_dim: int = 1024

    # 鼠标相关参数
    mouse_dim_in: int = 2
    mouse_hidden_dim: int = 1024
    mouse_qk_dim_list: List[int] = field(default_factory=lambda: [8, 28, 28])

    # Patch 和位置编码参数
    patch_size: List[int] = field(default_factory=lambda: [1, 2, 2])
    rope_dim_list: List[int] = field(default_factory=lambda: [8, 28, 28])  # 总和 = 64
    rope_theta: int = 256

    # QKV 参数
    qk_norm: bool = True
    qkv_bias: bool = False

    # 时间和窗口参数
    vae_time_compression_ratio: int = 4
    windows_size: int = 3
    local_attn_size: int = 6

    def __post_init__(self):
        """验证配置的有效性"""
        # 验证 rope_dim_list 总和（如果不匹配，会在运行时自动调整）
        head_dim = self.img_hidden_size // self.heads_num
        if sum(self.rope_dim_list) != head_dim:
            print(f"Warning: sum(rope_dim_list)={sum(self.rope_dim_list)} != head_dim={head_dim}")
            print(f"RoPE dimensions will be auto-adjusted at runtime to [{head_dim//3}, {head_dim//3}, {head_dim//3}]")

        # 验证 mouse_qk_dim_list 总和
        mouse_head_dim = self.mouse_hidden_dim // self.heads_num
        if sum(self.mouse_qk_dim_list) != mouse_head_dim:
            print(f"Warning: sum(mouse_qk_dim_list)={sum(self.mouse_qk_dim_list)} != mouse_head_dim={mouse_head_dim}")
            print(f"Mouse RoPE dimensions will be auto-adjusted at runtime")

        # 验证 patch_size 长度
        assert len(self.patch_size) == 3, "patch_size must have 3 elements [T, H, W]"

        # 验证至少启用一个模块
        if not self.enable_keyboard and not self.enable_mouse:
            print("Warning: Both keyboard and mouse are disabled!")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ActionConfig':
        """从字典创建配置"""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> 'ActionConfig':
        """从 JSON 文件加载配置"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_json(self, json_path: str, indent: int = 2):
        """保存为 JSON 文件"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    def __repr__(self) -> str:
        """美化打印"""
        lines = ["ActionConfig("]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value},")
        lines.append(")")
        return "\n".join(lines)

    @property
    def head_dim(self) -> int:
        """计算 attention head 的维度"""
        return self.img_hidden_size // self.heads_num

    @property
    def mouse_head_dim(self) -> int:
        """计算鼠标 attention head 的维度"""
        return self.mouse_hidden_dim // self.heads_num

    @property
    def keyboard_head_dim(self) -> int:
        """计算键盘 attention head 的维度"""
        return self.keyboard_hidden_dim // self.heads_num

    @property
    def is_enabled(self) -> bool:
        """检查是否启用任何 action 模块"""
        return self.enable_keyboard or self.enable_mouse


# 预定义的配置
DEFAULT_ACTION_CONFIG = ActionConfig()

SMALL_ACTION_CONFIG = ActionConfig(
    blocks=list(range(8)),
    heads_num=8,
    hidden_size=64,
    img_hidden_size=768,
    keyboard_hidden_dim=512,
    mouse_hidden_dim=512,
)

# 你提供的配置
WAN_1_3B_ACTION_CONFIG = ActionConfig(
    blocks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    enable_keyboard=True,
    enable_mouse=True,
    heads_num=16,
    hidden_size=128,
    img_hidden_size=1536,
    keyboard_dim_in=4,
    keyboard_hidden_dim=1024,
    mouse_dim_in=2,
    mouse_hidden_dim=1024,
    mouse_qk_dim_list=[8, 28, 28],
    patch_size=[1, 2, 2],
    qk_norm=True,
    qkv_bias=False,
    rope_dim_list=[8, 28, 28],
    rope_theta=256,
    vae_time_compression_ratio=4,
    windows_size=3,
)


def get_action_config(name: str = "default") -> ActionConfig:
    """
    获取预定义的配置

    Args:
        name: 配置名称，可选 "default", "small", "wan_1.3b"

    Returns:
        ActionConfig 实例
    """
    configs = {
        "default": DEFAULT_ACTION_CONFIG,
        "small": SMALL_ACTION_CONFIG,
        "wan_1.3b": WAN_1_3B_ACTION_CONFIG,
    }

    if name not in configs:
        raise ValueError(f"Unknown config name: {name}. Available: {list(configs.keys())}")

    return configs[name]


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Default Config:")
    print(DEFAULT_ACTION_CONFIG)

    print("\n" + "=" * 60)
    print("WAN 1.3B Config:")
    print(WAN_1_3B_ACTION_CONFIG)

    print("\n" + "=" * 60)
    print("Config properties:")
    config = WAN_1_3B_ACTION_CONFIG
    print(f"head_dim: {config.head_dim}")
    print(f"mouse_head_dim: {config.mouse_head_dim}")
    print(f"keyboard_head_dim: {config.keyboard_head_dim}")
    print(f"is_enabled: {config.is_enabled}")

    # 测试保存和加载
    print("\n" + "=" * 60)
    print("Test save/load:")
    config.to_json("/tmp/action_config.json")
    loaded_config = ActionConfig.from_json("/tmp/action_config.json")
    print(f"Configs equal: {config.to_dict() == loaded_config.to_dict()}")
