# scripts/test_grid_designer.py
import numpy as np

import core.constants as const
from core.models import BusModel, GridNetwork, LineModel, build_device_layout, device_of
from grid_model.builder import StubNetworkBuilder
from grid_model.environment import GridEnv
from grid_model.test_environment import _provider
from scenarios.policies import NaiveImmediatePolicy
from scenarios.runner import run_episode
from scripts.grid_designer import (
    StaticBuilder,
    available_checkpoints,
    bus_coordinates,
    household_color,
    layout_counts,
    nearest_household,
)


def _net_with_coords():
    buses = [BusModel(bus_id="h0", x_coord=8.40, y_coord=49.00),
             BusModel(bus_id="h1", x_coord=8.41, y_coord=49.01)]
    return GridNetwork(network_id="t", buses=buses, household_bus_ids=["h0", "h1"])


def _net_without_coords():
    buses = [BusModel(bus_id=f"h{i}") for i in range(6)]
    lines = [LineModel(line_id=f"l{i}", from_bus=f"h{i}", to_bus=f"h{i+1}",
                       length_km=0.1, r_ohm_per_km=0.3, x_ohm_per_km=0.08, max_i_ka=0.2)
             for i in range(5)]
    return GridNetwork(network_id="t", buses=buses, lines=lines,
                       household_bus_ids=[f"h{i}" for i in range(6)])


# ── coordinate handling ──────────────────────────────────────
def test_bus_coordinates_uses_real_coords_when_present():
    net = _net_with_coords()
    coords = bus_coordinates(net)
    assert coords["h0"] == (49.00, 8.40)   # (lat, lon)


def test_bus_coordinates_synthesizes_when_missing():
    net = _net_without_coords()
    coords = bus_coordinates(net)
    assert set(coords) == {b.bus_id for b in net.buses}
    assert all(isinstance(v, tuple) and len(v) == 2 for v in coords.values())


def test_nearest_household_picks_closest():
    hc = {"h0": (49.00, 8.40), "h1": (49.05, 8.50)}
    assert nearest_household(49.001, 8.401, hc) == "h0"
    assert nearest_household(49.049, 8.499, hc) == "h1"


# ── layout helpers ───────────────────────────────────────────
def test_layout_counts():
    layout = build_device_layout([f"h{i}" for i in range(10)], ev=0.6, battery=0.2, heat_pump=0.4, pv=0.8)
    counts = layout_counts(layout)
    assert counts[const.DEVICE_EV] == 6 and counts[const.DEVICE_BATTERY] == 2
    assert counts[const.DEVICE_HEAT_PUMP] == 4 and counts[const.DEVICE_PV] == 8


def test_household_color_by_device_count():
    from core.models import HouseholdDevices
    assert household_color(HouseholdDevices(bus_id="h")) == "#adb5bd"                       # 0
    assert household_color(HouseholdDevices(bus_id="h", ev=True, battery=True, heat_pump=True)) == "#e63946"  # 3


# ── end-to-end: custom layout → StaticBuilder → GridEnv → run ─
def test_static_builder_and_sim_run_with_custom_layout():
    net = StubNetworkBuilder(path="data/feeder_20.json").build()
    homes = list(net.household_bus_ids)
    layout = build_device_layout(homes, ev=0.5, battery=0.3, heat_pump=0.2, pv=0.7)
    env = GridEnv(builder=StaticBuilder(net), device_layout=layout, profile_provider=_provider())
    # agents reflect the custom per-device penetrations
    from collections import Counter
    c = Counter(device_of(a) for a in env.agent_ids)
    assert c[const.DEVICE_EV] == round(len(homes) * 0.5)
    assert c[const.DEVICE_BATTERY] == round(len(homes) * 0.3)
    # a full episode runs and produces device-power timelines
    result = run_episode(env, NaiveImmediatePolicy(), seed=0)
    assert len(result.timestep_results) == const.EPISODE_STEPS
    assert any(pf.device_power_kw.get(const.DEVICE_EV, 0.0) > 0 for pf in result.timestep_results)


def test_available_checkpoints(tmp_path):
    assert available_checkpoints(str(tmp_path)) == []
    (tmp_path / "pen_40").mkdir()
    (tmp_path / "pen_20").mkdir()
    (tmp_path / "not_a_ckpt").mkdir()
    assert available_checkpoints(str(tmp_path)) == ["pen_20", "pen_40"]
