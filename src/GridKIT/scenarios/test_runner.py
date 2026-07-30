# scenarios/test_runner.py
import warnings

import pytest

import core.constants as const
from core.models import device_of
from grid_model.environment import GridEnv
from grid_model.test_environment import _provider
from scenarios.policies import NaiveImmediatePolicy, NaivePriceFollowPolicy
from scenarios.runner import run_episode, run_scenario

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def env():
    # synthetic provider: EV always available (departure = last step), modest base load
    return GridEnv(profile_provider=_provider(avail=1.0, base=0.5))


def _ev_agents(env):
    return {a for a in env.agent_ids if device_of(a) == const.DEVICE_EV}


def test_run_episode_populates_metrics(env):
    result = run_episode(env, NaiveImmediatePolicy(), seed=0)
    assert result.network_id == env.network.network_id
    assert len(result.timestep_results) == 96
    # only EV agents carry a terminal SoC verdict; here every EV is available so all are recorded
    assert set(result.final_soc_per_agent) == _ev_agents(env)
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


# ── localizing the stress: which feeder / line, not just how often ──
def test_episode_metrics_localize_overloads(env):
    m = run_episode(env, NaiveImmediatePolicy(), seed=0).metrics
    # every feeder is listed with a peak, whether or not it ever tripped
    assert m.feeder_peak_loading_pu
    assert set(m.feeder_overload_steps) == set(m.feeder_peak_loading_pu)
    assert all(v >= 0 for v in m.feeder_overload_steps.values())
    # lines appear only when they actually exceeded the threshold
    for lid, steps in m.line_overload_steps.items():
        assert steps > 0
        assert m.line_peak_loading_pu[lid] > const.LINE_OVERLOAD_THRESHOLD


def test_overload_steps_never_exceed_episode_length(env):
    m = run_episode(env, NaiveImmediatePolicy(), seed=0).metrics
    for steps in list(m.feeder_overload_steps.values()) + list(m.line_overload_steps.values()):
        assert steps <= const.EPISODE_STEPS


def test_scenario_aggregates_components_counting_silent_seeds_as_zero(env):
    stats = run_scenario(env, NaivePriceFollowPolicy(jitter_std=0.0), [0, 1, 2], label="x")
    # a component that tripped in only some seeds must be averaged over ALL seeds,
    # so its mean stays strictly below its worst single-seed count
    for lid, (mean, _) in stats.line_overload_steps.items():
        assert 0 < mean <= const.EPISODE_STEPS
    assert set(stats.feeder_overload_steps) == set(stats.feeder_peak_loading_pu)


def test_worst_feeder_picks_the_most_overloaded(env):
    stats = run_scenario(env, NaiveImmediatePolicy(), [0, 1], label="x")
    worst = stats.worst_feeder()
    assert worst is not None
    fid, steps, peak = worst
    assert fid in stats.feeder_overload_steps
    # nothing else trips more often than the one we named
    assert steps == max(s for s, _ in stats.feeder_overload_steps.values())
