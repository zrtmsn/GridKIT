# rl_engine/test_trainer_timeouts.py
"""Unit tests for Trainer sample-timeout logic.

Covers:
- `_is_empty_iteration` — detecting the "sample timeout on every EnvRunner"
  case (0 steps, NaN episode-return mean) that otherwise yields silent
  empty/NaN training iterations.
- `_scaled_timeouts` — Option 3: timeouts grow with the grid-size factor f
  (timeout = basis × f).
- `_scaled_config_kwargs` — the scalings actually reach config_func when it
  accepts them.

No Ray cluster is started; these tests exercise pure Trainer logic only.
"""
import math

from GridKIT.rl_engine.trainer import Trainer, EmptySampleIterationError


def _trainer(config_func=None):
    return Trainer(env_factory=lambda: None, config_func=config_func or (lambda **kw: None))


# ── empty-iteration detection (sample timeout) ───────────────────────────

def test_is_empty_iteration_detects_all_timeout():
    """0 agent-steps / 0 episodes + NaN return == every runner timed out."""
    t = _trainer()
    empty = {
        "env_runners": {
            "num_agent_steps_sampled": 0,
            "num_episodes": 0,
            "episode_return_mean": float("nan"),
        }
    }
    assert t._is_empty_iteration(empty) is True


def test_is_empty_iteration_detects_missing_metric_fields():
    """A result without env_runners metrics at all counts as empty."""
    t = _trainer()
    assert t._is_empty_iteration({}) is True


def test_is_empty_iteration_normal_result_is_not_empty():
    """A healthy iteration (steps + episodes + finite return) is not empty."""
    t = _trainer()
    healthy = {
        "env_runners": {
            "num_agent_steps_sampled": 512,
            "num_episodes": 4,
            "episode_return_mean": -3865.35,
        }
    }
    assert t._is_empty_iteration(healthy) is False


def test_is_empty_iteration_zero_steps_but_finite_return_is_flagged():
    """Zero steps is always unusable, even if a stale return is present."""
    t = _trainer()
    stale = {
        "env_runners": {
            "num_agent_steps_sampled": 0,
            "num_episodes": 0,
            "episode_return_mean": 0.0,
        }
    }
    assert t._is_empty_iteration(stale) is True


# ── timeout scaling (Option 3: basis × f) ────────────────────────────────

def test_scaled_timeouts_is_noop_at_factor_1():
    t = _trainer()
    assert t._scaled_timeouts(1) == (None, None)


def test_scaled_timeouts_grow_with_factor():
    t = _trainer()
    assert t._scaled_timeouts(2) == (240.0, 480.0)
    assert t._scaled_timeouts(3) == (360.0, 720.0)


# ── scaling reaches config_func ──────────────────────────────────────────

def test_scaled_config_kwargs_passes_timeouts_when_accepted():
    received = {}

    def config_func(env_name, *, sample_timeout_s=None, evaluation_sample_timeout_s=None,
                    num_env_runners=None, train_batch_size=None, minibatch_size=None):
        received["sample_timeout_s"] = sample_timeout_s
        received["evaluation_sample_timeout_s"] = evaluation_sample_timeout_s
        return "dummy"

    t = _trainer(config_func)
    kwargs = t._scaled_config_kwargs(12)  # 12 agents, ref 6 → f=2
    assert kwargs["sample_timeout_s"] == 240.0
    assert kwargs["evaluation_sample_timeout_s"] == 480.0


def test_scaled_config_kwargs_omits_timeouts_when_not_accepted():
    t = _trainer(lambda env_name, **kw: "dummy")
    kwargs = t._scaled_config_kwargs(12)
    assert "sample_timeout_s" not in kwargs
    assert "evaluation_sample_timeout_s" not in kwargs


# ── exception type is importable/exported and fits the runtime error family ──

def test_empty_sample_iteration_error_is_runtime_error():
    assert issubclass(EmptySampleIterationError, RuntimeError)
