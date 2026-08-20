# core/test_models.py
import pytest

import core.constants as const
from core.models import (
    BatteryAction,
    BatteryState,
    DeviceType,
    HouseholdDevices,
    HPAction,
    HPState,
    Observation,
    build_device_layout,
)


# ── device action encodings line up with constants ───────────
def test_battery_action_indices_and_power_sign():
    assert (BatteryAction.DISCHARGE, BatteryAction.IDLE, BatteryAction.CHARGE) == (0, 1, 2)
    kw = const.BATTERY_ACTION_TO_KW
    assert kw[BatteryAction.DISCHARGE] < 0 < kw[BatteryAction.CHARGE]
    assert kw[BatteryAction.IDLE] == 0.0


def test_hp_action_dims_match_constants():
    assert (HPAction.OFF, HPAction.HEAT) == (0, 1)
    assert const.ACTION_DIM_BY_DEVICE[const.DEVICE_HEAT_PUMP] == len(HPAction)
    assert const.ACTION_DIM_BY_DEVICE[const.DEVICE_BATTERY] == len(BatteryAction)


def test_pv_is_not_a_controllable_agent():
    # PV folded into the battery decision — exogenous, no action, no policy
    assert const.DEVICE_PV not in const.CONTROLLABLE_DEVICE_TYPES
    assert const.DEVICE_PV not in const.ACTION_DIM_BY_DEVICE
    assert set(const.CONTROLLABLE_DEVICE_TYPES) == set(const.ACTION_DIM_BY_DEVICE)


def test_device_type_values_match_policy_suffixes():
    assert {d.value for d in DeviceType} == set(const.DEVICE_TYPES)
    assert set(const.CONTROLLABLE_DEVICE_TYPES) == {"ev", "battery", "hp"}


# ── state models ─────────────────────────────────────────────
def test_battery_state_defaults_and_bounds():
    b = BatteryState(agent_id="h1_battery", bus_id="h1")
    assert b.soc == const.BATTERY_INITIAL_SOC
    assert b.capacity_kwh == const.BATTERY_CAPACITY_KWH
    with pytest.raises(ValueError):
        BatteryState(agent_id="x", bus_id="h1", soc=1.5)


def test_hp_state_defaults():
    h = HPState(agent_id="h1_hp", bus_id="h1")
    assert h.thermal_soc == const.HP_INITIAL_THERMAL_SOC
    assert h.comfort_min_soc == const.HP_COMFORT_MIN_SOC


# ── observation: legacy vs multi-device layout ───────────────
def _obs(**kw):
    base = dict(
        agent_id="h1_ev", soc_progress=0.5, time_urgency=0.3,
        electricity_price=0.3, base_load_kw=0.5, outdoor_temperature_c=5.0,
    )
    base.update(kw)
    return Observation(**base)


def test_legacy_to_array_unchanged():
    o = _obs()
    assert len(o.to_array()) == const.OBS_DIM == 7
    # new fields are default-safe and do not leak into the legacy vector
    assert o.feed_in_price == 0.0 and o.device_type == DeviceType.EV


def test_multidevice_array_width_and_order():
    o = _obs(feed_in_price=0.08, pv_generation_kw=2.0, net_household_load_kw=1.5, time_of_day=0.25)
    arr = o.to_array_multidevice()
    assert len(arr) == const.OBS_DIM_MULTIDEVICE == 10
    assert arr[3] == 0.08 and arr[5] == 2.0 and arr[9] == 0.25


# ── per-household device layout ──────────────────────────────
def test_household_devices_controllable_order_excludes_pv():
    hd = HouseholdDevices(bus_id="h", ev=True, battery=False, heat_pump=True, pv=True)
    assert hd.controllable == [const.DEVICE_EV, const.DEVICE_HEAT_PUMP]  # order per CONTROLLABLE_DEVICE_TYPES
    assert const.DEVICE_PV not in hd.controllable                        # PV is exogenous, never controllable


def test_build_device_layout_per_device_penetrations():
    homes = [f"h{i}" for i in range(10)]
    layout = build_device_layout(homes, ev=0.6, battery=0.2, heat_pump=0.4, pv=0.8)
    assert set(layout) == set(homes)
    assert sum(c.ev for c in layout.values()) == 6
    assert sum(c.battery for c in layout.values()) == 2
    assert sum(c.heat_pump for c in layout.values()) == 4
    assert sum(c.pv for c in layout.values()) == 8


def test_build_device_layout_equal_fractions_is_all_or_nothing():
    # legacy joint behaviour: equal fractions → the SAME homes get the full stack
    homes = [f"h{i}" for i in range(10)]
    layout = build_device_layout(homes, ev=0.5, battery=0.5, heat_pump=0.5, pv=0.5)
    for cfg in layout.values():
        assert cfg.ev == cfg.battery == cfg.heat_pump == cfg.pv


def test_build_device_layout_zero_and_full():
    homes = [f"h{i}" for i in range(8)]
    none = build_device_layout(homes, ev=0.0, battery=0.0, heat_pump=0.0, pv=0.0)
    assert all(not c.controllable and not c.pv for c in none.values())
    full = build_device_layout(homes, ev=1.0, battery=1.0, heat_pump=1.0, pv=1.0)
    assert all(len(c.controllable) == 3 and c.pv for c in full.values())
