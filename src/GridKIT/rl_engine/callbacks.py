# rl_engine/callbacks.py
"""
Callbacks for RL training monitoring.
"""
from typing import Optional, Protocol
from dataclasses import dataclass


@dataclass
class TrainingResult:
    """Result container for a single training iteration."""
    iteration: int
    episode_return_mean: float
    episode_len_mean: float
    # Stamped on the LAST result of a training run so callers can tell whether
    # the loop ended because it converged or because it hit the max-iteration
    # ceiling.
    early_stopped: bool = False
    stop_reason: Optional[str] = None     # "converged" | "max_iterations" | None


class TrainingCallback(Protocol):
    """Protocol for training callbacks."""
    def on_iteration_end(self, iteration: int, reward: float, length: float) -> None:
        """Called after each training iteration."""
        ...


class DefaultCallback:
    """Default callback that prints progress."""
    def __init__(self, total: int):
        self.total = total

    def on_iteration_end(self, iteration: int, reward: float, length: float) -> None:
        print(f"  Iteration {iteration}/{self.total}: mean_reward={reward:.2f}, mean_len={length:.1f}")
