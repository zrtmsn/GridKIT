# rl_engine/trainer.py
"""
RLTrainer: Orchestrates RLlib training experiments.

Simplifies the training loop by encapsulating Ray/RLlib setup.
"""
from typing import Callable, List, Optional

import ray
from ray.tune.registry import register_env

from GridKIT.rl_engine.callbacks import TrainingCallback, DefaultCallback, TrainingResult
from GridKIT.core.config import settings


class Trainer:
    """
    High-level trainer that wraps Ray/RLlib complexity.

    Usage:
        # Minimal usage (uses DefaultCallback automatically)
        trainer = Trainer(env_factory=my_factory, config_func=create_ippo_config)
        results = trainer.run()  # prints progress via DefaultCallback

        # With custom callback
        trainer = Trainer(env_factory=my_factory, config_func=create_ippo_config)
        results = trainer.run(callback=my_callback)
    """

    def __init__(
        self,
        env_factory: Callable,
        config_func: Callable,
        env_name: str = settings.rllib_env_registry_name
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
        num_episodes: int = settings.rllib_default_num_episodes,
        callback: Optional[TrainingCallback] = None,
        cleanup: bool = True,
    ) -> List[TrainingResult]:
        """
        Run the training loop.

        Args:
            num_episodes: Number of training iterations.
            callback: Optional callback for progress updates.
            cleanup: If True, stop the algorithm and Ray when done. Pass False to
                keep the trained policy alive for evaluation (get_policy_module).

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

        if cleanup:
            self.stop()
        return results

    def get_policy_module(self, policy_id: str = "household_policy"):
        """Return the trained RLModule for inference (e.g. RLlibPolicyAdapter)."""
        if self._algo is None:
            raise RuntimeError("No trained algorithm — call run(cleanup=False) first.")
        return self._algo.get_module(policy_id)

    def save_checkpoint(self, path: str) -> str:
        """Persist the trained algorithm to `path`; returns the path."""
        if self._algo is None:
            raise RuntimeError("No trained algorithm to checkpoint.")
        self._algo.save_to_path(path)
        return path

    def stop(self):
        """Stop the algorithm and shut down Ray."""
        if self._algo is not None:
            self._algo.stop()
            self._algo = None
        if ray.is_initialized():
            ray.shutdown()

    # backward-compat alias
    _cleanup = stop