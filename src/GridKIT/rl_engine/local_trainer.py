# rl_engine/local_trainer.py
"""
LocalTrainer: Orchestrates RLlib training experiments for local development.

This module contains the complete trainer logic that was previously in trainer.py.
It is designed for local training on consumer grade hardware.
"""
from typing import Callable, List, Optional, Dict, Any

import ray
from ray.tune.registry import register_env

from GridKIT.rl_engine.callbacks import (
    TrainingCallback, 
    DefaultCallback, 
    TrainingResult,
)
from GridKIT.core import constants as const


class LocalTrainer:
    """
    High-level trainer for local development that wraps Ray/RLlib complexity.

    Usage:
        # Minimal usage (uses DefaultCallback automatically) 
        #TODO make this the headless usage
        trainer = LocalTrainer(env_factory=my_factory, config_func=create_ippo_config)
        results = trainer.run()  # prints progress via DefaultCallback

        # With custom callback
        trainer = LocalTrainer(env_factory=my_factory, config_func=create_ippo_config)
        results = trainer.run(callback=my_callback)

        #TODO comment on usage with dashboard-integration (e.g. no prints)
    """

    def __init__(
        self,
        env_factory: Callable,
        config_func: Callable,
        env_name: str = const.RLLIB_ENV_REGISTRY_NAME
    ):
        """
        Initialize the trainer.

        Args:
            env_factory: Callable that returns a new env instance per worker.
            config_func: Callable that returns a configured PPOConfig.
            env_name: Registered name of the environment.
        """
        self.env_factory = env_factory
        self.config_func = config_func
        self.env_name = env_name
        self._algo = None
        self.log_dir = None  # Will be set during training

    def _init_ray(self):
        """Initialize Ray if not already initialized."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _register_env(self):
        """Register the environment with Ray."""
        register_env(self.env_name, self.env_factory)

    def run(
        self,
        num_episodes: int = const.RLLIB_DEFAULT_NUM_EPISODES,
        callback: Optional[TrainingCallback] = None
    ) -> List[TrainingResult]:
        """
        Run the training loop.

        Args:
            num_episodes: Number of training iterations.
            callback: Optional callback for progress updates.

        Returns:
            List of TrainingResult objects, one per iteration.
        """
        self._init_ray()
        self._register_env()
        config = self.config_func(env_name=self.env_name)
        self._algo = config.build()


        #TODO feature: only print to console when in headless mode

        # Use default callback if none provided
        if callback is None:
            callback = DefaultCallback(total=num_episodes)

        # Collect raw RLlib result dicts during training
        results = []
        raw_rllib_results = []
        
        for i in range(num_episodes):
            result = self._algo.train()
            
            # Store raw RLlib result dict for later saving
            raw_rllib_results.append(result)

            training_result = TrainingResult(
                iteration=i + 1,
                episode_return_mean=result.get('env_runners', {}).get('episode_return_mean', 0.0),
                episode_len_mean=result.get('env_runners', {}).get('episode_len_mean', 0.0),
            )
            results.append(training_result)

            callback.on_iteration_end(
                training_result.iteration,
                training_result.episode_return_mean,
                training_result.episode_len_mean
            )
        
        # Save raw RLlib results to JSON file after training completes
        # Analogous to _write_step_to_file() in GridEnvRLlibWrapper
        if raw_rllib_results:
            self._save_raw_iteration_results(raw_rllib_results)

        self._cleanup()
        return results

    def _save_raw_iteration_results(self, raw_results: List[Dict[str, Any]]):
        """
        Save raw RLlib result dicts to JSON file.
        
        This is analogous to _write_step_to_file() in GridEnvRLlibWrapper:
        - Writes data directly without building Python objects
        - Called once at end of training (after all iterations complete)
        - Data will be loaded and transformed to objects by MetricsBuilder later
        
        Args:
            raw_results: List of raw RLlib result dicts from algo.train()
        """
        import json
        from pathlib import Path
        import tempfile
        
        self.log_dir = Path(tempfile.gettempdir()) / "gridkit_rl_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = self.log_dir / "iteration_metrics_raw.json"
        
        with open(output_file, 'w') as f:
            json.dump(raw_results, f, indent=2, default=str)
        
    
    def _cleanup(self):
        """Clean up resources."""
        if self._algo is not None:
            self._algo.stop()
        ray.shutdown()