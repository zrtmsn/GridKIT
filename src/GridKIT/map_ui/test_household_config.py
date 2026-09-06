# map_ui/test_household_config.py
import core.constants as const
from map_ui.household_config import (
    build_household_configuration,
    default_scenario_assumptions,
    device_layout_from_configuration,
    select_households_by_share,
)


class _Bounds:
    def model_dump(self):
        return {"south": 49.0, "west": 8.0, "north": 49.1, "east": 8.1}

    def approx_area_km2(self):
        return 1.0


class _Network:
    network_id = "test_net"
    area_name = "test_area"


def _config(ids, *, overrides=None, ev=50, battery=30, hp=20, pv=80, seed=42):
    return build_household_configuration(
        network=_Network(),
        selected_bounds=_Bounds(),
        household_ids=ids,
        ev_share_percent=ev,
        battery_share_percent=battery,
        heat_pump_share_percent=hp,
        pv_share_percent=pv,
        global_load_scaling_factor=1.0,
        selection_seed=seed,
        household_overrides=overrides or {},
    )


_IDS = [f"h{i}" for i in range(10)]


# ── all four device shares are configurable ──────────────────
def test_defaults_cover_all_four_devices():
    assumptions = default_scenario_assumptions()
    for field in ("ev_share_percent", "battery_share_percent",
                  "heat_pump_share_percent", "pv_share_percent"):
        assert field in assumptions


def test_shares_resolve_to_expected_counts():
    resolved = _config(_IDS)["resolved"]
    assert len(resolved["ev_bus_ids"]) == 5
    assert len(resolved["battery_bus_ids"]) == 3
    assert len(resolved["heat_pump_bus_ids"]) == 2
    assert len(resolved["pv_bus_ids"]) == 8


def test_zero_and_full_share_are_exact():
    resolved = _config(_IDS, ev=0, pv=100)["resolved"]
    assert resolved["ev_bus_ids"] == []
    assert resolved["pv_bus_ids"] == sorted(_IDS)


# ── per-device salts keep the draws independent ──────────────
def test_changing_one_share_does_not_reshuffle_another():
    # the reason each device gets its own salt: raising EV coverage must not
    # silently move which homes have a heat pump
    before = _config(_IDS, ev=20)["resolved"]["heat_pump_bus_ids"]
    after = _config(_IDS, ev=90)["resolved"]["heat_pump_bus_ids"]
    assert before == after


def test_same_seed_is_reproducible():
    assert _config(_IDS)["resolved"] == _config(_IDS)["resolved"]


def test_different_seed_changes_selection():
    assert _config(_IDS, seed=1)["resolved"] != _config(_IDS, seed=2)["resolved"]


def test_select_households_by_share_is_salt_sensitive():
    a = select_households_by_share(_IDS, 50, seed=42, salt="ev")
    b = select_households_by_share(_IDS, 50, seed=42, salt="battery")
    assert a != b


# ── per-household overrides, including the new battery/PV ────
def test_overrides_apply_for_every_device():
    overrides = {
        "h0": {"has_ev": True, "has_battery": True, "has_heat_pump": True, "has_pv": True},
        "h1": {"has_ev": False, "has_battery": False, "has_heat_pump": False, "has_pv": False},
    }
    resolved = _config(_IDS, overrides=overrides)["resolved"]
    for field in ("ev_bus_ids", "battery_bus_ids", "heat_pump_bus_ids", "pv_bus_ids"):
        assert "h0" in resolved[field]
        assert "h1" not in resolved[field]


def test_override_for_unknown_bus_is_ignored():
    cfg = _config(_IDS, overrides={"not_a_bus": {"has_ev": True}})
    assert "not_a_bus" not in cfg["individual_household_adjustments"]
    assert "not_a_bus" not in cfg["resolved"]["ev_bus_ids"]


def test_load_scaling_override_wins_over_global():
    cfg = _config(_IDS, overrides={"h2": {"load_scaling_factor": 1.5}})
    scaling = cfg["resolved"]["load_scaling_by_bus"]
    assert scaling["h2"] == 1.5
    assert scaling["h0"] == 1.0


# ── the bridge into core.models ──────────────────────────────
def test_device_layout_matches_resolved_sets():
    cfg = _config(_IDS)
    layout = device_layout_from_configuration(_IDS, cfg)
    resolved = cfg["resolved"]

    assert set(layout) == set(_IDS)
    for bus_id, devices in layout.items():
        assert devices.ev == (bus_id in resolved["ev_bus_ids"])
        assert devices.battery == (bus_id in resolved["battery_bus_ids"])
        assert devices.heat_pump == (bus_id in resolved["heat_pump_bus_ids"])
        assert devices.pv == (bus_id in resolved["pv_bus_ids"])


def test_layout_controllable_reflects_configuration():
    cfg = _config(_IDS, overrides={"h0": {"has_ev": True, "has_battery": True,
                                          "has_heat_pump": True}})
    layout = device_layout_from_configuration(_IDS, cfg)
    assert set(layout["h0"].controllable) == set(const.CONTROLLABLE_DEVICE_TYPES)


def test_pv_is_not_controllable():
    cfg = _config(_IDS, ev=0, battery=0, hp=0, pv=100)
    layout = device_layout_from_configuration(_IDS, cfg)
    assert layout["h0"].pv is True
    assert layout["h0"].controllable == []
