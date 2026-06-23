# scenarios/test_policies.py
import numpy as np

import core.constants as const
from core.models import ChargingAction, Observation
from scenarios.policies import NaiveImmediatePolicy, NaivePriceFollowPolicy


def _obs(agent_id, soc_progress):
    return Observation(
        agent_id=agent_id,
        soc_progress=soc_progress,
        time_urgency=0.5,
        electricity_price=0.3,
        base_load_kw=1.0,
        outdoor_temperature_c=20.0,
    )


def _flat_prices(value=0.3):
    return [value] * const.EPISODE_STEPS


# ── NaiveImmediatePolicy (scenario 1) ─────────────────────────

def test_immediate_charges_when_unfinished():
    p = NaiveImmediatePolicy()
    p.reset(_flat_prices(), np.random.default_rng(0))
    actions = p.act({"a": _obs("a", 0.5)})
    assert actions["a"] == ChargingAction.FULL


def test_immediate_stops_at_target():
    p = NaiveImmediatePolicy()
    p.reset(_flat_prices(), np.random.default_rng(0))
    actions = p.act({"a": _obs("a", 1.0)})
    assert actions["a"] == ChargingAction.OFF


# ── NaivePriceFollowPolicy (scenario 2) ───────────────────────

def test_price_follow_waits_for_cheap_window():
    # cheapest at step 50; everywhere else expensive
    prices = [1.0] * const.EPISODE_STEPS
    for s in range(48, 56):
        prices[s] = 0.05
    p = NaivePriceFollowPolicy(jitter_std=0.0)
    p.reset(prices, np.random.default_rng(0))

    obs = {"a": _obs("a", 0.95)}   # small remaining need → short window near the trough
    actions_over_time = [p.act(obs)["a"] for _ in range(const.EPISODE_STEPS)]

    # step function: OFF early, FULL once the chosen window starts, never FULL-before-OFF
    assert actions_over_time[0] == ChargingAction.OFF
    assert actions_over_time[-1] == ChargingAction.FULL
    first_full = actions_over_time.index(ChargingAction.FULL)
    assert 44 <= first_full <= 56
    assert all(a == ChargingAction.OFF for a in actions_over_time[:first_full])


def test_price_follow_jitter_zero_is_deterministic():
    prices = _flat_prices()
    prices[60] = 0.01
    starts = []
    for _ in range(3):
        p = NaivePriceFollowPolicy(jitter_std=0.0)
        p.reset(prices, np.random.default_rng(0))
        p.act({"a": _obs("a", 0.9)})
        starts.append(p._start_step["a"])
    assert len(set(starts)) == 1


def test_price_follow_jitter_spreads_starts():
    prices = _flat_prices()
    prices[60] = 0.01
    # many agents, wide jitter → starts should not collapse to a single value
    obs = {f"a{i}": _obs(f"a{i}", 0.5) for i in range(20)}
    p = NaivePriceFollowPolicy(jitter_std=8.0)
    p.reset(prices, np.random.default_rng(1))
    p.act(obs)
    assert len(set(p._start_step.values())) > 1
