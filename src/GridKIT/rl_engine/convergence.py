# rl_engine/convergence.py
"""Plateau-based convergence detection for RL training runs.

Pure logic — no Ray/RLlib imports — so it is unit-testable in isolation and
shared by every training entry point (trainer.py, and through it train_run.py
and run_experiment.py).

Rationale: the number of training iterations should not be a fixed constant
independent of the network size. Instead the loop runs until the reward has
converged (a plateau), with min/max bounds as the safety net. Because the batch
sizes grow with the agent count (see trainer._scaled_batch_sizes), the same
plateau criterion is meaningful for a 5-household stub and a 500-household
grid.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from GridKIT.core.constants import PIPELINE_MAX_TRAINING_ITERATIONS
from GridKIT.core.config import settings


@dataclass
class StopDecision:
    """Outcome of one convergence check."""
    stop: bool
    reason: Optional[str] = None   # "converged" | "max_iterations" | None


class ConvergenceTracker:
    """Plateau-based early stopping on episode_return_mean.

    Stops when the mean episode return has not improved by at least `min_delta`
    over the last `patience` iterations, but never before `min_iterations`
    iterations have run (protects against stopping on an unlucky first score).
    `max_iterations` is the hard ceiling: even without convergence the run may
    not continue beyond it.

    Reward traces are noisy, so this deliberately tracks the *best* reward seen
    so far — one unlucky bad iteration does not consume patience — and only
    counts an improvement once it exceeds `min_delta`, so a plateau of noise is
    not mistaken for progress.

    Smoothing: with `smooth_window = k > 1` the plateau logic operates on the
    rolling average of the LAST `k` per-iteration reward means — the window
    slides, it is NOT the mean of all rewards since the start, and already
    averaged values are NOT averaged a second time. Until the window is full the
    available values are used (a partial mean). A single lucky noise spike is
    damped by the window, so the stop point no longer depends on when the noise
    happens to peak. `smooth_window = 1` disables smoothing (raw values).
    """

    def __init__(
        self,
        patience: int = settings.pipeline_early_stop_patience,
        min_delta: float = settings.ippo_early_stop_min_delta,
        min_iterations: int = settings.pipeline_min_training_iterations,
        max_iterations: int = PIPELINE_MAX_TRAINING_ITERATIONS,
        smooth_window: int = settings.pipeline_early_stop_smooth_window,
    ):
        # Never let patience/min_iterations degenerate into an immediate stop.
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.min_iterations = max(0, int(min_iterations))
        self.max_iterations = max(1, int(max_iterations))
        self.smooth_window = max(1, int(smooth_window))
        self._best_reward = float("-inf")
        self._best_iteration = 0
        self._converged = False
        # Sliding window of the last `smooth_window` raw per-iteration means.
        self._recent_rewards: deque[float] = deque(maxlen=self.smooth_window)
        self._smoothed: Optional[float] = None

    @property
    def converged(self) -> bool:
        """True once the plateau criterion has fired."""
        return self._converged

    @property
    def smoothed_reward(self) -> Optional[float]:
        """Latest sliding-window mean the plateau logic operates on (None before
        the first iteration is fed)."""
        return self._smoothed

    def should_stop(self, iteration: int, episode_return_mean: float) -> StopDecision:
        """Check whether training should stop after the given iteration.

        Args:
            iteration: 1-based training iteration number.
            episode_return_mean: mean episode reward reported for that iteration.

        Returns:
            StopDecision(stop=True, reason=...) when the run should stop
            ("converged" for the plateau, "max_iterations" for the ceiling),
            StopDecision(stop=False) otherwise.
        """
        # Smooth the raw per-iteration means over a sliding window: the plateau
        # logic compares the mean of the LAST `smooth_window` values (partial
        # while the window is still filling up), never the average of everything
        # seen so far — and never a re-averaging of already averaged values.
        raw_reward = float(episode_return_mean)
        self._recent_rewards.append(raw_reward)
        self._smoothed = sum(self._recent_rewards) / len(self._recent_rewards)
        reward = self._smoothed

        # An improvement beyond min_delta resets the patience clock.
        if reward > self._best_reward + self.min_delta:
            self._best_reward = reward
            self._best_iteration = iteration

        # Plateau criterion — only once the minimum run length is reached.
        if (
            not self._converged
            and iteration >= self.min_iterations
            and iteration - self._best_iteration >= self.patience
        ):
            self._converged = True
            return StopDecision(stop=True, reason="converged")

        # Hard ceiling.
        if iteration >= self.max_iterations:
            return StopDecision(stop=True, reason="max_iterations")

        return StopDecision(stop=False)
