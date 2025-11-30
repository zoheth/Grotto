"""
Modular Action Module Package

Terminology:
    - View Control: Camera/view angle control (input: mouse, gamepad right stick)
    - Movement: Character movement (input: keyboard WASD, gamepad left stick)
"""

from grotto.modeling.modular_action.action_config import ActionConfig
from grotto.modeling.modular_action.action_module import ActionModule, ActionContext
from grotto.modeling.modular_action.view_control import (
    ViewControlInjector,
    ViewControlPreprocessor,
)
from grotto.modeling.modular_action.movement_control import (
    MovementInjector,
    MovementPreprocessor,
)
from grotto.modeling.modular_action.interfaces import (
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
