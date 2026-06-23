# scenarios/test_runner.py
import warnings

import pytest

from grid_model.environment import GridEnv
from scenarios.policies import NaiveImmediatePolicy, NaivePriceFollowPolicy
from scenarios.runner import run_episode, run_scenario

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def env():
    return GridEnv()


def test_run_episode_populates_metrics(env):
    result = run_episode(env, NaiveImmediatePolicy(), seed=0)
    assert result.network_id == env.network.network_id
    assert len(result.timestep_results) == 96
    # every agent terminates within the episode → final SoC recorded for all
    assert set(result.final_soc_per_agent) == set(env.agent_ids)
    m = result.metrics
    assert m.curtailment_events >= 0
    assert 0.0 <= m.soc_satisfaction_rate <= 1.0


def test_run_scenario_aggregates_over_seeds(env):
    stats = run_scenario(env, NaiveImmediatePolicy(), seeds=[0, 1], label="t")
    assert stats.n_episodes == 2
    for pair in (stats.curtailment_events, stats.soc_satisfaction_rate,
                 stats.transformer_peak_loading_pu):
        assert len(pair) == 2  # (mean, std)


def test_automation_increases_curtailment(env):
    """σ→0 (automated) should not curtail less than wide-σ (manual) — the core mechanism."""
    seeds = [0, 1, 2]
    manual = run_scenario(env, NaivePriceFollowPolicy(jitter_std=8.0), seeds, label="manual")
    automated = run_scenario(env, NaivePriceFollowPolicy(jitter_std=0.0), seeds, label="auto")
    assert automated.curtailment_events[0] >= manual.curtailment_events[0]
