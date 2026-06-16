# rl_engine/rl_control.py
"""
RL Control: Orchestrator for RL training in GridKIT.

This module is the central entry point for starting RL training. It selects
the appropriate trainer (local vs HPC) and coordinates post-training metrics
collection.

Usage:
    from GridKIT.rl_engine.rl_control import RLControl, TrainerType
    
    # Create control with local trainer
    control = RLControl(trainer_type=TrainerType.LOCAL)
    
"""

from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from GridKIT.core.models import AllMetrics
from GridKIT.core.config import settings


class TrainerType(Enum):
    """Enum for selecting trainer type."""
    LOCAL = "local"  # For local development
    HPC = "hpc"      # For HPC clusters (stub)


class RLControl:
    """
    Central orchestrator for RL training.
    
    This class:
    1. Selects the appropriate trainer (local or HPC)
    2. Runs the training loop
    3. Triggers post-training metrics collection
    4. Returns AllMetrics container
    
    Args:
        trainer_type: Which trainer to use (LOCAL or HPC)
        log_dir: Optional custom log directory
    """
    
    def __init__(
        self,
        trainer_type: TrainerType = TrainerType.LOCAL,
        log_dir: Optional[Path] = None
    ):
        self.trainer_type = trainer_type
        self.log_dir = log_dir
        self._trainer = None
    
    def _get_trainer(self, env_factory: Callable, config_func: Callable, env_name: str):
        """Get the appropriate trainer based on trainer_type."""
        if self.trainer_type == TrainerType.LOCAL:
            from GridKIT.rl_engine.local_trainer import LocalTrainer
            return LocalTrainer(
                env_factory=env_factory,
                config_func=config_func,
                env_name=env_name,
            )
        else:
            # HPC trainer stub - for future implementation
            raise NotImplementedError(
                "HPC trainer is not yet implemented. Make sure to use TrainerType.LOCAL for now."
            )
    
    def run_training(
        self,
        env_factory: Callable,
        config_func: Callable,
        num_iterations: int = settings.rllib_default_num_iterations,
        env_name: str = settings.rllib_env_registry_name,
    ) -> AllMetrics:
        """
        Run RL training and return all metrics.
        
        Args:
            env_factory: Callable that returns new env instance per worker
            config_func: Callable that returns configured PPOConfig
            num_episodes: Number of training iterations
            env_name: Registered name of the environment
            
        Returns:
            AllMetrics container with episode_metrics, sim_results, and iteration_metrics
        """
        # Get trainer
        self._trainer = self._get_trainer(env_factory, config_func, env_name)
        
        # Run training with default callback for console output
        from GridKIT.rl_engine.callbacks import DefaultCallback
        callback = DefaultCallback(total=num_iterations)
        self._trainer.run(num_iterations=num_iterations, callback=callback)
        
        # Post-training: Build all metrics
        from GridKIT.rl_engine.metrics_builder import MetricsBuilder
        
        builder = MetricsBuilder(log_dir=self._trainer.log_dir)
        return builder.build_all()
    
    def get_trainer_log_dir(self) -> Optional[Path]:
        """Get the log directory from the trainer after training."""
        if self._trainer is None:
            return None
        return getattr(self._trainer, 'log_dir', None)