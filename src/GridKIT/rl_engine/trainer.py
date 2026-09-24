# rl_engine/trainer.py
"""
RLTrainer: Orchestrates RLlib training experiments.

Simplifies the training loop by encapsulating Ray/RLlib setup.
"""
import inspect
import math
import os
from typing import Callable, List, Optional

import ray
from ray.tune.registry import register_env

from GridKIT.rl_engine.callbacks import TrainingCallback, DefaultCallback, TrainingResult
from GridKIT.rl_engine.convergence import ConvergenceTracker
from GridKIT.core.constants import PIPELINE_MAX_TRAINING_ITERATIONS
from GridKIT.core.config import settings


class EmptySampleIterationError(RuntimeError):
    """Raised when a training iteration collected no data at all.

    RLlib discards EnvRunner results that exceed `sample_timeout_s`; when every
    runner times out the iteration is empty (0 steps, NaN return mean) and no
    learning update can be performed. In practice this means the grid is too
    large for the available hardware (see Trainer._is_empty_iteration).
    """


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

    def _count_agents(self) -> int:
        """Number of controllable-device agents in the training environment.

        Builds one throwaway env instance from `env_factory` and reads its agent
        list (`possible_agents` on GridEnvRLlibWrapper, `agent_ids` on a bare
        GridEnv). Returns 0 if the count cannot be determined — in that case
        `run()` falls back to the unscaled defaults.
        """
        env = None
        try:
            env = self.env_factory()
            for attr in ("possible_agents", "agent_ids"):
                ids = getattr(env, attr, None)
                if ids:
                    return len(ids)
            return 0
        except Exception:
            return 0
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _scaling_factor(self, n_agents: int) -> int:
        """Common scaling factor for batch sizes and env runners.

        ceil(n_agents / settings.ippo_batch_scaling_ref_agents), at least 1.
        The floor at 1 means the scaling only ever GROWS the batch — a network
        with fewer agents than the reference grid (the 6-agent minimal stub the
        512 batch was tuned for) keeps the tuned defaults unchanged instead of
        shrinking the batch below one episode's worth of data.

        Shared by _scaled_batch_sizes() and _scaled_num_env_runners() so both
        grow with the same factor as the grid.
        """
        if n_agents <= 0:
            return 1
        return max(1, math.ceil(n_agents / settings.ippo_batch_scaling_ref_agents))

    def _scaled_batch_sizes(self, n_agents: int) -> tuple[Optional[int], Optional[int]]:
        """Scale train_batch_size / minibatch_size with the agent count.

        `train_batch_size` counts agent-steps SUMMED over all agents. With a
        fixed 512 and more agents, each agent contributes fewer transitions per
        update. Scaling by the agent-count factor keeps the transitions
        PER AGENT per update constant across grid sizes, so the number of
        training iterations needed to converge does not grow with the grid.

        Returns (train_batch_size, minibatch_size); both None when scaling is a
        no-op (n_agents <= ref_agents) so the configured defaults pass through.
        """
        factor = self._scaling_factor(n_agents)
        if factor == 1:
            return None, None
        train_batch_size = settings.ippo_train_batch_size * factor
        minibatch_size = min(settings.ippo_minibatch_size * factor, train_batch_size)
        return train_batch_size, minibatch_size

    def _scaled_num_env_runners(self, factor: int) -> Optional[int]:
        """Scale the number of parallel env workers with the agent count.

        With the batch scaled by `factor`, each iteration must collect `factor`
        times as many agent-steps, and each simulated step is itself more
        expensive (bigger grid). Scaling the worker count by the same factor
        keeps the collection time per iteration roughly constant.

        Worker count:
        - Preferred value: half the machine's cores, floored
          (os.cpu_count() // 2), but never below the configured default
          (max(base, ...)) — the other half of the cores stays free for the
          learner and the OS; more workers than that only adds
          process-communication overhead.

        Returns None when scaling is a no-op (factor 1, or capped at the
        default) so the configured default is passed through unchanged.
        """
        if factor <= 1:
            return None
        base = settings.ippo_num_env_runners
        scaled = base * factor
        cpus = os.cpu_count() or 0
        if cpus > 1:
            scaled = max(base, cpus // 2)
        if scaled == base:
            return None
        return scaled

    def _scaled_timeouts(self, factor: int) -> tuple[Optional[float], Optional[float]]:
        """Scale the sample timeouts with the agent count.

        A bigger grid takes longer per EnvRunner sample() call (more agents,
        more expensive steps). RLlib DISCARDS results that take longer than
        `sample_timeout_s`, which yields empty iterations (0 steps, NaN return,
        no update). So the timeouts grow proportionally with the same factor f
        as the batch sizes:  timeout = basis × f.

        Returns (sample_timeout_s, evaluation_sample_timeout_s); both None
        when scaling is a no-op (f == 1) so the configured defaults pass
        through unchanged.
        """
        if factor <= 1:
            return None, None
        return (
            settings.ippo_sample_timeout_s * factor,
            settings.ippo_evaluation_sample_timeout_s * factor,
        )

    @staticmethod
    def _is_empty_iteration(result: dict) -> bool:
        """True if an RLlib iteration result contained no sampled data.

        A sample-timeout (`config.sample_timeout_s`) makes RLlib discard the
        results of EnvRunners that took too long — when nothing comes back the
        iteration has 0 steps and a NaN episode-return mean, so no learning
        update happens. Detected on the same metric fields the dashboard reads.
        """
        env_r = result.get("env_runners") or {}
        steps = env_r.get("num_agent_steps_sampled", 0)
        episodes = env_r.get("num_episodes", 0)
        return_mean = env_r.get("episode_return_mean", None)
        nan_return = return_mean is None or (
            isinstance(return_mean, float) and math.isnan(return_mean)
        )
        # 0 steps/episodes is always an empty iteration; a NaN (or missing)
        # return mean with data present is just as unusable for an update.
        return steps == 0 or episodes == 0 or nan_return

    def _scaled_config_kwargs(self, n_agents: int) -> dict:
        """kwargs for config_func that scale with the agent count, or {}.

        Only passes a value through when config_func can actually accept it
        (checked via inspect.signature) AND the scaled value differs from the
        configured default — a config_func that takes just ``env_name`` keeps
        working untouched.
        """
        if n_agents <= 0:
            return {}
        params = inspect.signature(self.config_func).parameters
        scaled: dict = {}
        factor = self._scaling_factor(n_agents)
        train_batch_size, minibatch_size = self._scaled_batch_sizes(n_agents)
        num_env_runners = self._scaled_num_env_runners(factor)
        sample_timeout_s, evaluation_sample_timeout_s = self._scaled_timeouts(factor)
        if "train_batch_size" in params and train_batch_size is not None:
            scaled["train_batch_size"] = train_batch_size
        if "minibatch_size" in params and minibatch_size is not None:
            scaled["minibatch_size"] = minibatch_size
        if "num_env_runners" in params and num_env_runners is not None:
            scaled["num_env_runners"] = num_env_runners
        if "sample_timeout_s" in params and sample_timeout_s is not None:
            scaled["sample_timeout_s"] = sample_timeout_s
        if "evaluation_sample_timeout_s" in params and evaluation_sample_timeout_s is not None:
            scaled["evaluation_sample_timeout_s"] = evaluation_sample_timeout_s
        if scaled:
            # ASCII only, deliberately. This runs in a detached subprocess whose
            # stdout is a redirected file, so its encoding comes from the locale.
            # No legacy Windows codepage (cp1250, cp1252, cp850) can encode an
            # arrow, and a log line must never be able to kill the run.
            print(
                f"[Trainer] {n_agents} agents, scaled train_batch_size "
                f"{settings.ippo_train_batch_size} -> {scaled.get('train_batch_size', settings.ippo_train_batch_size)}, "
                f"minibatch_size {settings.ippo_minibatch_size} -> "
                f"{scaled.get('minibatch_size', settings.ippo_minibatch_size)}, "
                f"num_env_runners {settings.ippo_num_env_runners} -> "
                f"{scaled.get('num_env_runners', settings.ippo_num_env_runners)}, "
                f"sample_timeout_s {settings.ippo_sample_timeout_s} -> "
                f"{scaled.get('sample_timeout_s', settings.ippo_sample_timeout_s)}, "
                f"evaluation_sample_timeout_s {settings.ippo_evaluation_sample_timeout_s} -> "
                f"{scaled.get('evaluation_sample_timeout_s', settings.ippo_evaluation_sample_timeout_s)}"
            )
        return scaled

    def run(
        self,
        max_iterations: int = PIPELINE_MAX_TRAINING_ITERATIONS,
        callback: Optional[TrainingCallback] = None,
        cleanup: bool = True,
        metrics_dir=None,
        patience: int = settings.pipeline_early_stop_patience,
        min_delta: float = settings.ippo_early_stop_min_delta,
        min_iterations: int = settings.pipeline_min_training_iterations,
        smooth_window: int = settings.pipeline_early_stop_smooth_window,
    ) -> List[TrainingResult]:
        """
        Run the training loop, stopping once the reward has converged.

        The run length is NOT a fixed constant: `max_iterations` is only the hard
        ceiling. Training stops early when the mean episode return has stopped
        improving (plateau, `patience`/`min_delta` — see convergence.py), so a
        small grid converges in a few iterations while a large user-configured
        grid is given the iterations it actually needs. Batch sizes are scaled
        with the grid's agent count (see _scaled_batch_sizes), keeping the
        transitions PER AGENT per update — and therefore the effect of one
        iteration — constant across network sizes.

        Args:
            max_iterations: MAXIMUM number of training iterations (hard ceiling).
            callback: Optional callback for progress updates.
            cleanup: If True, stop the algorithm and Ray when done. Pass False to
                keep the trained policy alive for evaluation (get_policy_module).
            metrics_dir: Where to write `iteration_metrics.json`. Pass the run's own
                directory so several runs (e.g. one per penetration level) each keep
                their own metrics — without it they all land on the same temp path
                and overwrite each other.
            patience: Stop after this many iterations without an improvement of
                at least `min_delta` (only once `min_iterations` have run).
            min_delta: Minimum improvement of episode_return_mean that still
                counts as progress.
            min_iterations: Never early-stop before this many iterations have
                run (protects against stopping on an unlucky first score).
            smooth_window: Sliding window `k` over the per-iteration reward
                means fed to the convergence check. The plateau logic operates
                on the mean of the last k values (partial until the window is
                full) — NOT the mean since the start, and not a re-average.
                k=1 disables smoothing.

        Returns:
            List of TrainingResult objects, one per iteration. The last element
            carries `early_stopped`/`stop_reason` ("converged" or
            "max_iterations") so callers can tell why training ended.
        """
        self._init_ray()
        self._register_env()

        # Scale batch sizes / env runners with the grid's agent count so the
        # value of one iteration stays constant across user-configured network
        # sizes (see _scaled_batch_sizes).
        n_agents = self._count_agents()
        config_kwargs = self._scaled_config_kwargs(n_agents)
        config_kwargs["env_name"] = self.env_name
        config = self.config_func(**config_kwargs)
        self._algo = config.build()

        # Use default callback if none provided
        if callback is None:
            callback = DefaultCallback(total=max_iterations)

        # Plateau-based convergence detector (never stops before min_iterations,
        # never longer than max_iterations). It compares the smoothed
        # sliding-window reward (see convergence.py) against the best seen.
        tracker = ConvergenceTracker(
            patience=patience,
            min_delta=min_delta,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            smooth_window=smooth_window,
        )

        # Collect raw RLlib result dicts during training
        results = []
        raw_rllib_results = []
        early_stopped = False
        stop_reason: Optional[str] = None

        for i in range(max_iterations):
            result = self._algo.train()

            # A sample-timeout on every EnvRunner discards all data → empty
            # iteration (0 steps, NaN return, no update). Abort immediately
            # instead of silently "training" on nothing.
            if self._is_empty_iteration(result):
                if cleanup:
                    self.stop()
                raise EmptySampleIterationError(
                    "Training iteration collected no samples (sample_timeout_s "
                    "exceeded by every EnvRunner) — the grid is too large for "
                    "the available hardware."
                )

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

            # Convergence check: stop once the reward plateaued (convergence.py).
            decision = tracker.should_stop(i + 1, training_result.episode_return_mean)
            if decision.stop:
                early_stopped = decision.reason == "converged"
                stop_reason = decision.reason
                break

        if stop_reason is None:
            stop_reason = "max_iterations"
        # Stamped on the last result so callers (train_run.py's status message,
        # the return value) can tell a converged run from one that ran its full
        # budget.
        if results:
            results[-1].early_stopped = early_stopped
            results[-1].stop_reason = stop_reason
        print(f"[Trainer] Training finished after {len(results)}/{max_iterations} iterations "
              f"(reason: {stop_reason})")

        # Save raw RLlib results to JSON file after training completes
        if raw_rllib_results:
            self._save_raw_iteration_results(raw_rllib_results, metrics_dir)

        if cleanup:
            self.stop()
        return results
    
    def _save_raw_iteration_results(self, raw_results: List[dict], metrics_dir=None) -> None:
        """
        Save raw RLlib result dicts to JSON file.

        Analogous to logging in GridEnvRLlibWrapper:
        - Writes data directly without building Python objects
        - Called once at end of training (after all iterations complete)
        - Data will be loaded and transformed to objects by MetricsBuilder later
        
        Args:
            raw_results: List of raw RLlib result dicts from algo.train()
        """
        import json
        from pathlib import Path
        import tempfile

        if metrics_dir is not None:
            log_dir = Path(metrics_dir)
            # the name the dashboard and DATENSTRUKTUR_DOKUMENTATION.md expect
            filename = "iteration_metrics.json"
        else:
            log_dir = Path(tempfile.gettempdir()) / "gridkit_rl_logs"
            filename = "iteration_metrics_raw.json"

        log_dir.mkdir(parents=True, exist_ok=True)
        output_file = log_dir / filename

        with open(output_file, 'w') as f:
            json.dump(raw_results, f, indent=2, default=str)

    def get_policy_module(self, policy_id: str):
        """Return one trained RLModule by policy id (e.g. 'ev_policy')."""
        if self._algo is None:
            raise RuntimeError("No trained algorithm — call run(cleanup=False) first.")
        return self._algo.get_module(policy_id)

    def get_policy_modules(self) -> dict:
        """Return {device_type -> trained RLModule} for all shared per-device policies
        (feed straight into RLlibPolicyAdapter)."""
        from GridKIT.core import constants as const
        return {dev: self.get_policy_module(f"{dev}_policy")
                for dev in const.CONTROLLABLE_DEVICE_TYPES}

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