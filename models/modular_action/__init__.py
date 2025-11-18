"""
Modular Action Module Package
"""

from .action_config import (
    ActionConfig,
    DEFAULT_ACTION_CONFIG,
    SMALL_ACTION_CONFIG,
    WAN_1_3B_ACTION_CONFIG,
    get_action_config,
)

from .action_module import ActionModule

from .injectors import (
    MouseInjector,
    KeyboardInjector,
)

from .interfaces import (
    IAttentionInjector,
    IActionPreprocessor,
    KVCacheManager,
    FlashInferAttentionCore,
    WanRMSNorm,
)

__all__ = [
    # Main module
    "ActionModule",

    # Configuration
    "ActionConfig",
    "DEFAULT_ACTION_CONFIG",
    "SMALL_ACTION_CONFIG",
    "WAN_1_3B_ACTION_CONFIG",
    "get_action_config",

    # Injectors
    "MouseInjector",
    "KeyboardInjector",

    # Interfaces
    "IAttentionInjector",
    "IActionPreprocessor",
    "KVCacheManager",
    "FlashInferAttentionCore",
    "WanRMSNorm",
]
