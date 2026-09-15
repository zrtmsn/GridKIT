# grid_model/test_environment.py
import numpy as np
import pytest

import core.constants as const
from core.models import (
    BatteryAction,
    ChargingAction,
    HouseholdDevices,
    HPAction,
    Observation,
    PowerFlowResult,
    StepResult,
)
from grid_model.device_profiles import DeviceProfileProvider, _ANNUAL_HOURS
from grid_model.environment import (
    GridEnv,
    battery_step,
    curtail_household,
    device_of,
    hp_step,
    make_agent_id,
)


# ── synthetic provider (constant profiles → deterministic, no pyCity) ──
def _provider(avail=1.0, base=0.3, pv=0.0, hp=0.0, temp=10.0, n_sims=2):
    ones = np.ones(_ANNUAL_HOURS)
    return DeviceProfileProvider(
        base_load_sims=np.full((n_sims, _ANNUAL_HOURS), base),
        ev_avail_sims=np.full((n_sims, _ANNUAL_HOURS), avail),
        hp_annual=ones * hp,
        pv_annual_per_kwp=ones * (pv / 5.0),   # scaled back up by a sampled kWp
        temp_annual=ones * temp,
        pv_kwp_range=(5.0, 5.0),
    )


def _env(provider=None, **kw):
    return GridEnv(profile_provider=provider or _provider(), **kw)


def _actions(env, ev=0, batt=BatteryAction.IDLE, hp=HPAction.OFF):
    a = {}
    for aid in env.agent_ids:
        dev = device_of(aid)
        a[aid] = {const.DEVICE_EV: int(ev), const.DEVICE_BATTERY: int(batt),
                  const.DEVICE_HEAT_PUMP: int(hp)}[dev]
    return a


# ── agents / topology ────────────────────────────────────────
def test_three_agents_per_active_household():
    env = _env()
    n_house = len(env._active_bus_ids)
    assert len(env.agent_ids) == 3 * n_house
    assert {device_of(a) for a in env.agent_ids} == set(const.CONTROLLABLE_DEVICE_TYPES)


def test_penetration_limits_active_households():
    assert len(_env(ev_penetration=0.4).agent_ids) < len(_env(ev_penetration=1.0).agent_ids)


# ── reset ────────────────────────────────────────────────────
def test_reset_returns_observation_per_agent():
    env = _env()
    obs = env.reset(seed=0)
    assert set(obs) == set(env.agent_ids)
    assert all(isinstance(o, Observation) for o in obs.values())
    assert env.current_step == 0


def test_reset_reproducible_with_seed():
    a = _env().reset(seed=3)
    b = _env().reset(seed=3)
    k = next(iter(a))
    assert a[k].outdoor_temperature_c == b[k].outdoor_temperature_c


def test_observation_multidevice_width():
    env = _env()
    o = next(iter(env.reset(seed=0).values()))
    assert len(o.to_array_multidevice()) == const.OBS_DIM_MULTIDEVICE
    assert o.feed_in_price == const.FEED_IN_TARIFF_EUR_KWH


# ── step mechanics ───────────────────────────────────────────
def test_step_types_and_agent_coverage():
    env = _env()
    env.reset(seed=0)
    results, pf = env.step(_actions(env))
    assert isinstance(pf, PowerFlowResult)
    assert set(results) == set(env.agent_ids)
    assert all(isinstance(r, StepResult) for r in results.values())


def test_step_increments_and_done_at_last():
    env = _env()
    env.reset(seed=0)
    for _ in range(const.EPISODE_STEPS - 1):
        results, _ = env.step(_actions(env))
    assert all(not r.done for r in results.values())
    results, _ = env.step(_actions(env))
    assert all(r.done for r in results.values())


def test_household_agents_share_reward():
    env = _env()
    env.reset(seed=0)
    results, _ = env.step(_actions(env, ev=ChargingAction.FULL))
    bus = env._active_bus_ids[0]
    rewards = {device_of(a): results[a].reward for a in env.agent_ids if a.startswith(bus + "::")}
    assert len(set(rewards.values())) == 1


def test_ev_full_charges_when_connected():
    env = _env(_provider(avail=1.0))
    env.reset(seed=0)
    bus = env._active_bus_ids[0]
    soc0 = env._households[bus].ev.soc
    env.step(_actions(env, ev=ChargingAction.FULL))
    assert env._households[bus].ev.soc > soc0


def test_ev_does_not_charge_when_disconnected():
    env = _env(_provider(avail=0.0))
    env.reset(seed=0)
    bus = env._active_bus_ids[0]
    soc0 = env._households[bus].ev.soc
    env.step(_actions(env, ev=ChargingAction.FULL))
    assert env._households[bus].ev.soc == soc0


def test_battery_charge_raises_soc_discharge_lowers():
    env = _env(_provider())
    env.reset(seed=0)
    bus = env._active_bus_ids[0]
    s0 = env._households[bus].battery.soc
    env.step(_actions(env, batt=BatteryAction.CHARGE))
    s1 = env._households[bus].battery.soc
    assert s1 > s0
    env.step(_actions(env, batt=BatteryAction.DISCHARGE))
    assert env._households[bus].battery.soc < s1


def test_battery_uses_custom_capacity_from_household_devices():
    # a HouseholdDevices.battery_kwh override (e.g. GridCreator's real
    # per-household `storage` value) must reach BatteryState.capacity_kwh,
    # not just the fixed BATTERY_CAPACITY_KWH constant.
    from grid_model.builder import StubNetworkBuilder

    stub_net = StubNetworkBuilder().build()
    bus = stub_net.household_bus_ids[0]
    custom_kwh = 3.0
    layout = {bus: HouseholdDevices(bus_id=bus, battery=True, battery_kwh=custom_kwh)}

    env = _env(device_layout=layout)
    env.reset(seed=0)

    assert env._households[bus].battery.capacity_kwh == custom_kwh

    # Same physics, smaller denominator → a smaller battery fills faster at
    # the same charge power.
    dt = const.TIMESTEP_MINUTES / 60
    soc_custom, _ = battery_step(0.5, BatteryAction.CHARGE, capacity_kwh=custom_kwh,
                                  max_power_kw=const.BATTERY_MAX_POWER_KW,
                                  efficiency=const.BATTERY_EFFICIENCY, dt=dt)
    soc_default, _ = battery_step(0.5, BatteryAction.CHARGE, capacity_kwh=const.BATTERY_CAPACITY_KWH,
                                   max_power_kw=const.BATTERY_MAX_POWER_KW,
                                   efficiency=const.BATTERY_EFFICIENCY, dt=dt)
    assert soc_custom > soc_default


def test_pv_export_earns_feed_in_positive_reward():
    # large PV, no consumption → household exports → bill negative → reward > 0
    env = _env(_provider(base=0.0, pv=8.0))
    env.reset(seed=0)
    results, _ = env.step(_actions(env))
    assert next(iter(results.values())).reward > 0


# ── §14a curtailment ─────────────────────────────────────────
def test_synchronized_full_load_triggers_curtailment():
    env = _env(_provider(avail=1.0, base=0.5))
    env.reset(seed=0)
    _, pf = env.step(_actions(env, ev=ChargingAction.FULL, batt=BatteryAction.CHARGE, hp=HPAction.HEAT))
    assert pf.curtailment_applied
    assert pf.curtailed_power_kw


def test_no_curtailment_when_idle():
    env = _env(_provider(avail=1.0, base=0.1))
    env.reset(seed=0)
    _, pf = env.step(_actions(env))
    assert not pf.curtailment_applied


def test_curtailment_is_localized_to_the_overloaded_feeder():
    from core.models import BusModel, GridNetwork, LineModel, TransformerModel, build_device_layout
    # feeder A: tiny 0.02 MVA trafo (will overload); feeder B: huge 1.0 MVA (stays fine)
    buses = [BusModel(bus_id=b) for b in ("hvA", "lvA", "h0", "h1", "hvB", "lvB", "h2", "h3")]
    lines = [LineModel(line_id=f"l{i}", from_bus=a, to_bus=b, length_km=0.02,
                       r_ohm_per_km=0.6, x_ohm_per_km=0.08, max_i_ka=0.5)   # generous lines → trafo is the limit
             for i, (a, b) in enumerate([("lvA", "h0"), ("h0", "h1"), ("lvB", "h2"), ("h2", "h3")])]
    trafos = [TransformerModel(trafo_id="tA", hv_bus="hvA", lv_bus="lvA", s_nom_mva=0.02),
              TransformerModel(trafo_id="tB", hv_bus="hvB", lv_bus="lvB", s_nom_mva=1.0)]
    net = GridNetwork(network_id="2f", buses=buses, lines=lines, transformers=trafos,
                      household_bus_ids=["h0", "h1", "h2", "h3"])
    layout = build_device_layout(net.household_bus_ids, ev=1.0, battery=1.0, heat_pump=1.0, pv=0.0)

    env = GridEnv(builder=type("B", (), {"build": lambda s: net})(), device_layout=layout,
                  profile_provider=_provider(avail=1.0, base=0.3))
    env.reset(seed=0)
    _, pf = env.step(_actions(env, ev=ChargingAction.FULL, batt=BatteryAction.CHARGE, hp=HPAction.HEAT))

    assert pf.curtailment_applied
    assert pf.transformer_loadings_pu["lvA"] > 1.0        # feeder A overloaded
    assert pf.transformer_loadings_pu["lvB"] < 1.0        # feeder B fine
    # only feeder-A households were dimmed
    assert "h0" in pf.curtailed_power_kw and "h1" in pf.curtailed_power_kw
    assert "h2" not in pf.curtailed_power_kw and "h3" not in pf.curtailed_power_kw


# ── pure device dynamics ─────────────────────────────────────
def test_battery_step_respects_soc_window():
    soc, bus = battery_step(const.BATTERY_MAX_SOC, BatteryAction.CHARGE, capacity_kwh=10, max_power_kw=5,
                            efficiency=0.95, dt=0.25)
    assert soc <= const.BATTERY_MAX_SOC + 1e-9 and bus == pytest.approx(0.0, abs=1e-6)
    soc, bus = battery_step(const.BATTERY_MIN_SOC, BatteryAction.DISCHARGE, capacity_kwh=10, max_power_kw=5,
                            efficiency=0.95, dt=0.25)
    assert soc >= const.BATTERY_MIN_SOC - 1e-9 and bus == pytest.approx(0.0, abs=1e-6)


def test_battery_charge_sign_and_efficiency():
    soc, bus = battery_step(0.5, BatteryAction.CHARGE, capacity_kwh=10, max_power_kw=5,
                            efficiency=0.9, dt=1.0)
    assert bus > 0
    assert soc == pytest.approx(0.5 + 5 * 0.9 / 10)


def test_hp_heat_charges_buffer_off_drains():
    up, elec = hp_step(0.5, HPAction.HEAT, demand_thermal_kw=0.0, cop=3.0, capacity_kwh=8.0, dt=0.25)
    assert up > 0.5 and elec == const.HP_RATED_ELECTRIC_KW
    down, elec = hp_step(0.5, HPAction.OFF, demand_thermal_kw=6.0, cop=3.0, capacity_kwh=8.0, dt=0.25)
    assert down < 0.5 and elec == 0.0


def test_curtail_household_floor_and_scaling():
    assert curtail_household(10.0, 0.5) == pytest.approx(5.0)
    assert curtail_household(10.0, 0.1) == pytest.approx(const.MIN_GUARANTEED_POWER_KW)
    assert curtail_household(2.0, 0.1) == pytest.approx(2.0)


def test_day_ahead_prices_length():
    env = _env()
    env.reset(seed=0)
    assert len(env.day_ahead_prices()) == const.EPISODE_STEPS


# ── EV SoC is scored proportionally, not as a cliff ──────────
def _terminal_ev_infos(policy_charges: bool):
    """Run one episode charging (or never charging) and return the EV terminal infos."""
    from core.models import ChargingAction
    env = _env(provider=_provider(avail=1.0, base=0.3))
    obs = env.reset(seed=0)
    infos = []
    while True:
        actions = {}
        for aid in obs:
            if device_of(aid) == const.DEVICE_EV:
                actions[aid] = int(ChargingAction.FULL if policy_charges else ChargingAction.OFF)
            else:
                actions[aid] = 0
        step_results, _ = env.step(actions)
        for aid, r in step_results.items():
            if r.info.get("ev_terminal"):
                infos.append(r.info)
        obs = {aid: r.observation for aid, r in step_results.items()}
        if any(r.done for r in step_results.values()):
            break
    return infos


def test_shortfall_is_recorded_and_bounded():
    infos = _terminal_ev_infos(policy_charges=False)
    assert infos, "expected at least one EV to reach its departure step"
    for i in infos:
        assert 0.0 <= i["ev_shortfall"] <= 1.0


def test_a_car_that_charges_has_no_shortfall():
    for i in _terminal_ev_infos(policy_charges=True):
        if i["ev_satisfied"]:
            assert i["ev_shortfall"] == 0.0


def _soc_penalty(shortfall: float) -> float:
    """The terminal EV term as environment._reward applies it."""
    return const.REWARD_SOC_MISS_PENALTY * shortfall ** 2


#: Bill spread between the cheapest and most expensive strategy, measured on the
#: 245-home ding0 grid (€16.89 immediate vs €12.32). This is the prize the agent is
#: trading against, so the SoC term has to be calibrated relative to it.
_ACHIEVABLE_DAILY_SAVING = 4.5


def test_missing_by_a_hair_is_nearly_free():
    # a car 2% short has range margin to absorb it; scoring that as a failure is what
    # drove the agent to ignore price entirely under the old cliff
    assert abs(_soc_penalty(0.02)) < 0.5


def test_price_still_steers_the_policy_at_small_shortfalls():
    # if even a tiny deviation costs more than a day's achievable saving, price becomes
    # irrelevant and the agent degenerates to "charge immediately"
    assert abs(_soc_penalty(0.05)) < _ACHIEVABLE_DAILY_SAVING


def test_deep_undercharging_is_prohibitive():
    # the failure this replaced: a LINEAR -25 let the agent sell 14% of the charge for a
    # cheaper bill. Gutting half the charge must cost far more than any saving available.
    assert abs(_soc_penalty(0.5)) > 10 * _ACHIEVABLE_DAILY_SAVING


def test_penalty_is_convex_not_linear():
    # doubling the shortfall must more than double the penalty, or it stays exploitable
    assert abs(_soc_penalty(0.4)) > 2 * abs(_soc_penalty(0.2))
