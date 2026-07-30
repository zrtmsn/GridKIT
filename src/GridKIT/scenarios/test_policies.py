# scenarios/test_policies.py
import numpy as np

import core.constants as const
from core.models import BatteryAction, ChargingAction, HPAction, Observation, make_agent_id
from scenarios.policies import (
    NaiveImmediatePolicy,
    NaivePriceFollowPolicy,
    _battery_greedy,
    _hp_thermostatic,
)


def _obs(device, soc_progress, *, pv=0.0, net_load=0.0, bus="bus_0"):
    return Observation(
        agent_id=make_agent_id(bus, device),
        soc_progress=soc_progress,
        time_urgency=0.5,
        electricity_price=0.3,
        base_load_kw=1.0,
        outdoor_temperature_c=20.0,
        pv_generation_kw=pv,
        net_household_load_kw=net_load,
    )


def _flat_prices(value=0.3):
    return [value] * const.EPISODE_STEPS


# ── NaiveImmediatePolicy (scenario 1) — EV leg ────────────────
def test_immediate_charges_when_unfinished():
    p = NaiveImmediatePolicy()
    p.reset(_flat_prices(), np.random.default_rng(0))
    a = p.act({make_agent_id("bus_0", const.DEVICE_EV): _obs(const.DEVICE_EV, 0.5)})
    assert a[make_agent_id("bus_0", const.DEVICE_EV)] == int(ChargingAction.FULL)


def test_immediate_stops_at_target():
    p = NaiveImmediatePolicy()
    p.reset(_flat_prices(), np.random.default_rng(0))
    a = p.act({make_agent_id("bus_0", const.DEVICE_EV): _obs(const.DEVICE_EV, 1.0)})
    assert a[make_agent_id("bus_0", const.DEVICE_EV)] == int(ChargingAction.OFF)


# ── NaivePriceFollowPolicy (scenario 2) — EV leg ──────────────
def test_price_follow_waits_for_cheap_window():
    prices = [1.0] * const.EPISODE_STEPS
    for s in range(48, 56):
        prices[s] = 0.05
    p = NaivePriceFollowPolicy(jitter_std=0.0)
    p.reset(prices, np.random.default_rng(0))
    aid = make_agent_id("bus_0", const.DEVICE_EV)
    obs = {aid: _obs(const.DEVICE_EV, 0.95)}
    over_time = [p.act(obs)[aid] for _ in range(const.EPISODE_STEPS)]
    assert over_time[0] == int(ChargingAction.OFF)
    assert over_time[-1] == int(ChargingAction.FULL)
    first_full = over_time.index(int(ChargingAction.FULL))
    assert 44 <= first_full <= 56
    assert all(a == int(ChargingAction.OFF) for a in over_time[:first_full])


def test_price_follow_jitter_zero_is_deterministic():
    prices = _flat_prices()
    prices[60] = 0.01
    aid = make_agent_id("bus_0", const.DEVICE_EV)
    starts = []
    for _ in range(3):
        p = NaivePriceFollowPolicy(jitter_std=0.0)
        p.reset(prices, np.random.default_rng(0))
        p.act({aid: _obs(const.DEVICE_EV, 0.9)})
        starts.append(p._start_step[aid])
    assert len(set(starts)) == 1


def test_price_follow_jitter_spreads_starts():
    prices = _flat_prices()
    prices[60] = 0.01
    obs = {make_agent_id(f"bus_{i}", const.DEVICE_EV): _obs(const.DEVICE_EV, 0.5, bus=f"bus_{i}")
           for i in range(20)}
    p = NaivePriceFollowPolicy(jitter_std=8.0)
    p.reset(prices, np.random.default_rng(1))
    p.act(obs)
    assert len(set(p._start_step.values())) > 1


# ── fixed device controllers (shared by both baselines) ───────
def test_battery_greedy_charges_on_pv_surplus():
    assert _battery_greedy(_obs(const.DEVICE_BATTERY, 0.5, pv=2.0)) == int(BatteryAction.CHARGE)


def test_battery_greedy_discharges_on_import():
    assert _battery_greedy(_obs(const.DEVICE_BATTERY, 0.5, pv=0.0, net_load=2.0)) == int(BatteryAction.DISCHARGE)


def test_battery_greedy_idles_when_balanced():
    assert _battery_greedy(_obs(const.DEVICE_BATTERY, 0.5, pv=0.0, net_load=0.0)) == int(BatteryAction.IDLE)


def test_battery_greedy_respects_soc_limits():
    # full battery does not keep charging even with PV surplus
    assert _battery_greedy(_obs(const.DEVICE_BATTERY, const.BATTERY_MAX_SOC, pv=2.0)) != int(BatteryAction.CHARGE)


def test_hp_thermostatic_heats_below_setpoint_off_above():
    assert _hp_thermostatic(_obs(const.DEVICE_HEAT_PUMP, 0.3)) == int(HPAction.HEAT)
    assert _hp_thermostatic(_obs(const.DEVICE_HEAT_PUMP, 0.9)) == int(HPAction.OFF)
