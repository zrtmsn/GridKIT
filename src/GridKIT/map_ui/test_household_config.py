# map_ui/test_household_config.py
from types import SimpleNamespace

from core.models import HouseholdDevices
from map_ui.household_config import build_household_configuration


class _Bounds:
    def model_dump(self):
        return {}

    def approx_area_km2(self):
        return 1.0


def _network(household_devices=None):
    return SimpleNamespace(
        network_id="n", area_name="test-area",
        household_devices=household_devices or {},
    )


def _build(network, household_ids, overrides=None, ev_share=30, hp_share=20):
    return build_household_configuration(
        network=network,
        selected_bounds=_Bounds(),
        household_ids=household_ids,
        ev_share_percent=ev_share,
        heat_pump_share_percent=hp_share,
        global_load_scaling_factor=1.0,
        selection_seed=42,
        household_overrides=overrides or {},
    )


def test_defaults_from_gridcreator_when_present():
    household_ids = ["h0", "h1"]
    devices = {
        "h0": HouseholdDevices(bus_id="h0", ev=True, heat_pump=False, battery=True, pv=True, pv_kwp=6.5, battery_kwh=12.0),
        "h1": HouseholdDevices(bus_id="h1", ev=False, heat_pump=True, battery=False, pv=False),
    }
    config = _build(_network(devices), household_ids)

    resolved = config["resolved"]
    assert resolved["ev_bus_ids"] == ["h0"]
    assert resolved["heat_pump_bus_ids"] == ["h1"]
    assert resolved["battery_bus_ids"] == ["h0"]
    assert resolved["pv_bus_ids"] == ["h0"]
    assert resolved["pv_kwp_by_bus"] == {"h0": 6.5}
    assert resolved["battery_kwh_by_bus"] == {"h0": 12.0}
    assert config["scenario_assumptions"]["uses_gridcreator_defaults"] is True


def test_falls_back_to_share_percent_when_no_gridcreator_data():
    household_ids = [f"h{i}" for i in range(10)]
    config = _build(_network(), household_ids, ev_share=50, hp_share=0)

    resolved = config["resolved"]
    assert len(resolved["ev_bus_ids"]) == 5   # 50% of 10, deterministic given the seed
    assert resolved["heat_pump_bus_ids"] == []
    # no share concept for battery/pv — stay off without GridCreator data
    assert resolved["battery_bus_ids"] == []
    assert resolved["pv_bus_ids"] == []
    assert config["scenario_assumptions"]["uses_gridcreator_defaults"] is False


def test_manual_override_wins_over_gridcreator_default():
    household_ids = ["h0"]
    devices = {"h0": HouseholdDevices(bus_id="h0", ev=True, battery=True)}
    config = _build(
        _network(devices), household_ids,
        overrides={"h0": {"has_ev": False, "has_battery": False, "has_pv": True}},
    )

    resolved = config["resolved"]
    assert resolved["ev_bus_ids"] == []
    assert resolved["battery_bus_ids"] == []
    assert resolved["pv_bus_ids"] == ["h0"]


def test_manual_override_wins_over_share_percent_fallback():
    household_ids = ["h0", "h1"]
    config = _build(
        _network(), household_ids, ev_share=0,
        overrides={"h0": {"has_ev": True}},
    )

    assert config["resolved"]["ev_bus_ids"] == ["h0"]
