"""
Modular Action Module Package

Terminology:
    - View Control: Camera/view angle control (input: mouse, gamepad right stick)
    - Movement: Character movement (input: keyboard WASD, gamepad left stick)
"""

from .action_config import (
    ActionConfig
)

from .action_module import ActionModule, ActionContext

from .view_control import (
    ViewControlInjector,
    ViewControlPreprocessor,
)
from .movement_control import (
    MovementInjector,
    MovementPreprocessor,
)

from .interfaces import (
    ActionInjector,
    AttentionKernel
)

__all__ = [
    # Main module
    "ActionModule",
    "ActionContext",

    # Configuration
    "ActionConfig",

    # Injectors
    "ViewControlInjector",
    "MovementInjector",
    "ViewControlPreprocessor",
    "MovementPreprocessor",

    # Core components
    "ActionInjector",
    "AttentionKernel",
]
