"""
Modular Action Module Package

Terminology:
    - View Control: Camera/view angle control (input: mouse, gamepad right stick)
    - Movement: Character movement (input: keyboard WASD, gamepad left stick)
"""

from grotto.modeling.modular_action.action_config import ActionConfig
from grotto.modeling.modular_action.action_module import ActionContext, ActionModule
from grotto.modeling.modular_action.interfaces import ActionInjector, AttentionKernel
from grotto.modeling.modular_action.movement_control import (
    MovementInjector,
    MovementPreprocessor,
)
from grotto.modeling.modular_action.view_control import (
    ViewControlInjector,
    ViewControlPreprocessor,
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
