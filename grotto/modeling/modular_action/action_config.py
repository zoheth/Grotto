import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ActionConfig:
    blocks: List[int] = field(default_factory=lambda: list(range(15)))

    enable_keyboard: bool = True
    enable_mouse: bool = True

    heads_num: int = 16
    hidden_size: int = 128
    img_hidden_size: int = 1024

    keyboard_dim_in: int = 4
    keyboard_hidden_dim: int = 1024

    mouse_dim_in: int = 2
    mouse_hidden_dim: int = 1024
    mouse_qk_dim_list: List[int] = field(default_factory=lambda: [8, 28, 28])

    patch_size: List[int] = field(default_factory=lambda: [1, 2, 2])
    rope_dim_list: List[int] = field(default_factory=lambda: [8, 28, 28])
    rope_theta: int = 256

    qk_norm: bool = True
    qkv_bias: bool = False

    vae_time_compression_ratio: int = 4
    windows_size: int = 3
    local_attn_size: int = 6

    def __post_init__(self):
        head_dim = self.img_hidden_size // self.heads_num
        if sum(self.rope_dim_list) != head_dim:
            print(f"Warning: sum(rope_dim_list)={sum(self.rope_dim_list)} != head_dim={head_dim}")
            print(
                f"RoPE dimensions will be auto-adjusted at runtime to [{head_dim // 3}, {head_dim // 3}, {head_dim // 3}]"
            )

        mouse_head_dim = self.mouse_hidden_dim // self.heads_num
        if sum(self.mouse_qk_dim_list) != mouse_head_dim:
            print(
                f"Warning: sum(mouse_qk_dim_list)={sum(self.mouse_qk_dim_list)} != mouse_head_dim={mouse_head_dim}"
            )
            print("Mouse RoPE dimensions will be auto-adjusted at runtime")

        assert len(self.patch_size) == 3, "patch_size must have 3 elements [T, H, W]"

        if not self.enable_keyboard and not self.enable_mouse:
            print("Warning: Both keyboard and mouse are disabled!")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ActionConfig":
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "ActionConfig":
        with open(json_path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, json_path: str, indent: int = 2):
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    def __repr__(self) -> str:
        lines = ["ActionConfig("]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value},")
        lines.append(")")
        return "\n".join(lines)

    @property
    def head_dim(self) -> int:
        return self.img_hidden_size // self.heads_num

    @property
    def mouse_head_dim(self) -> int:
        return self.mouse_hidden_dim // self.heads_num

    @property
    def keyboard_head_dim(self) -> int:
        return self.keyboard_hidden_dim // self.heads_num

    @property
    def is_enabled(self) -> bool:
        return self.enable_keyboard or self.enable_mouse
