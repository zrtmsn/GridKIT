# rl_engine/trainer.py
"""
RLTrainer: Orchestrates RLlib training experiments.

Simplifies the training loop by encapsulating Ray/RLlib setup.
"""
from typing import Callable, List, Optional

import ray
from ray.tune.registry import register_env

from GridKIT.rl_engine.callbacks import TrainingCallback, DefaultCallback, TrainingResult


class RLTrainer:
    """
    High-level trainer that wraps Ray/RLlib complexity.

    Usage:
        callback = DefaultCallback(total=10)
        trainer = RLTrainer(env_factory=my_factory, config_func=my_config)
        results = trainer.run(num_episodes=10, callback=callback)
    """

    def __init__(
        self,
        env_factory: Callable,
        config_func: Callable,
        env_name: str = "GridEnv-v0"
    ):
        """
        Initialize the trainer.

        Args:
            env_factory: Callable that returns a new env instance per worker.
            config_func: Callable that returns a configured PPOConfig.
            env_name: Registered name for the environment.
        """
        self.env_factory = env_factory
        self.config_func = config_func
        self.env_name = env_name
        self._algo = None

    def _init_ray(self):
        """Initialize Ray if not already initialized."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

    def _register_env(self):
        """Register the environment with Ray."""
        register_env(self.env_name, self.env_factory)

    def run(
        self,
        num_episodes: int = 10,
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

        # Use default callback if none provided
        if callback is None:
            callback = DefaultCallback(total=num_episodes)

        results = []
        for i in range(num_episodes):
            result = self._algo.train()

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

        self._cleanup()
        return results

    def _cleanup(self):
        """Clean up resources."""
        if self._algo is not None:
            self._algo.stop()
        ray.shutdown()