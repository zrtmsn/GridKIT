# rl_engine/__init__.py
"""
RL-Engine Module for GridKIT.
Provides interfaces for training agents via RLlib.

IMPORTANT: Nothing is executed automatically here!
Use factory classes to initialize components.
"""

# Export only the classes/functions intended for external use.

from .grid_env_rllib_wrapper import GridEnvRLlibWrapper

#TODO Future extensions (e.g., a dedicated trainer wrapper or config helper):
#TODO from .trainer import IPPOTrainer

__all__ = [
    "GridEnvRLlibWrapper",
    # "IPPOTrainer",
]


