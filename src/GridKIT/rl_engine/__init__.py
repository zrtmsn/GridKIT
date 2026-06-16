# rl_engine/__init__.py
"""
RL-Engine Module for GridKIT.
Provides interfaces for training agents via RLlib.

IMPORTANT: Nothing is executed automatically here!
Use factory classes to initialize components.
"""

# Export only the classes/functions intended for external use.

from .grid_env_rllib_wrapper import GridEnvRLlibWrapper
from .ippo_config import create_ippo_config
from .callbacks import TrainingCallback, DefaultCallback, TrainingResult

# New orchestrator
from .rl_control import RLControl, TrainerType

# Metrics builder (post-training)
from .metrics_builder import (
    build_episode_metrics_from_file,
    build_sim_result_from_file,
    build_iteration_metrics_from_rl_result,
    EpisodeMetricsCollector,
    MetricsBuilder,
)

__all__ = [
    # Core
    "GridEnvRLlibWrapper",
    "create_ippo_config",
    
    # Callbacks
    "TrainingCallback",
    "DefaultCallback",
    "TrainingResult",
    
    # Orchestrator 
    "RLControl",
    "TrainerType",
    
    # Metrics Builder 
    "build_episode_metrics_from_file",
    "build_sim_result_from_file",
    "build_iteration_metrics_from_rl_result", 
    "EpisodeMetricsCollector",
    "MetricsBuilder",
]
