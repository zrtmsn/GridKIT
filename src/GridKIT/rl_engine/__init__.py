# rl_engine/__init__.py
"""
RL-Engine Module for GridKIT.
Provides interfaces for training agents via RLlib.

IMPORTANT: Nothing is executed automatically here!
Use factory classes to initialize components.
"""

# Export only the classes/functions intended for external use.

from .grid_env_rllib_wrapper import GridEnvRLlibWrapper
from .trainer import Trainer, EmptySampleIterationError
from .ippo_config import create_ippo_config
from .callbacks import TrainingCallback, DefaultCallback, TrainingResult
from .rl_policy import RLlibPolicyAdapter

__all__ = [
    "GridEnvRLlibWrapper",
    "Trainer",
    "EmptySampleIterationError",
    "create_ippo_config",
    "TrainingCallback",
    "DefaultCallback",
    "TrainingResult",
    "RLlibPolicyAdapter",
]


