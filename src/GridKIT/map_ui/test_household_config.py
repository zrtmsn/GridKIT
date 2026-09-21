from types import SimpleNamespace

from core.models import HouseholdDevices
from map_ui.household_config import build_household_configuration


class _Bounds:
    def model_dump(self):
        return {
            "south": 49.0,
            "west": 8.4,
            "north": 49.01,
            "east": 8.41,
        }

    def approx_area_km2(self):
        return 1.0


def _network(
    household_devices=None,
    household_load_profile_kw=None,
    ev_availability=None,
):
    return SimpleNamespace(
        network_id="n",
        area_name="test-area",
        household_devices=household_devices or {},
        household_load_profile_kw=household_load_profile_kw or {},
        ev_availability=ev_availability or {},
    )


def _build(
    network,
    household_ids,
    overrides=None,
    ev_share=30.0,
    hp_share=20.0,
    load_scaling=1.0,
    seed=42,
):
    return build_household_configuration(
        network=network,
        selected_bounds=_Bounds(),
        household_ids=household_ids,
        ev_share_percent=ev_share,
        heat_pump_share_percent=hp_share,
        global_load_scaling_factor=load_scaling,
        selection_seed=seed,
        household_overrides=overrides or {},
    )


def test_uses_real_household_device_data_when_present():
    household_ids = ["h0", "h1"]

    devices = {
        "h0": HouseholdDevices(
            bus_id="h0",
            ev=True,
            heat_pump=False,
            battery=True,
            pv=True,
            pv_kwp=6.5,
            battery_kwh=12.0,
        ),
        "h1": HouseholdDevices(
            bus_id="h1",
            ev=False,
            heat_pump=True,
            battery=False,
            pv=False,
        ),
    }

    config = _build(_network(household_devices=devices), household_ids)

    resolved = config["resolved"]

    assert resolved["ev_bus_ids"] == ["h0"]
    assert resolved["heat_pump_bus_ids"] == ["h1"]
    assert resolved["battery_bus_ids"] == ["h0"]
    assert resolved["pv_bus_ids"] == ["h0"]
    assert resolved["pv_kwp_by_bus"] == {"h0": 6.5}
    assert resolved["battery_kwh_by_bus"] == {"h0": 12.0}
    assert config["scenario_assumptions"]["uses_gridcreator_defaults"] is True


def test_supports_dictionary_device_data():
    household_ids = ["h0"]

    devices = {
        "h0": {
            "ev": True,
            "heat_pump": True,
            "battery": True,
            "pv": True,
            "pv_kwp": 7.5,
            "battery_kwh": 10.0,
        }
    }

    config = _build(_network(household_devices=devices), household_ids)

    resolved = config["resolved"]

    assert resolved["ev_bus_ids"] == ["h0"]
    assert resolved["heat_pump_bus_ids"] == ["h0"]
    assert resolved["battery_bus_ids"] == ["h0"]
    assert resolved["pv_bus_ids"] == ["h0"]
    assert resolved["pv_kwp_by_bus"] == {"h0": 7.5}
    assert resolved["battery_kwh_by_bus"] == {"h0": 10.0}


def test_uses_share_percent_when_no_real_device_data_exists():
    household_ids = [f"h{i}" for i in range(10)]

    config = _build(
        _network(),
        household_ids,
        ev_share=50.0,
        hp_share=0.0,
    )

    resolved = config["resolved"]

    assert len(resolved["ev_bus_ids"]) == 5
    assert resolved["heat_pump_bus_ids"] == []
    assert resolved["battery_bus_ids"] == []
    assert resolved["pv_bus_ids"] == []
    assert config["scenario_assumptions"]["uses_gridcreator_defaults"] is False


def test_manual_override_wins_over_real_device_data():
    household_ids = ["h0"]

    devices = {
        "h0": HouseholdDevices(
            bus_id="h0",
            ev=True,
            heat_pump=False,
            battery=True,
            pv=False,
        )
    }

    config = _build(
        _network(household_devices=devices),
        household_ids,
        overrides={
            "h0": {
                "has_ev": False,
                "has_battery": False,
                "has_pv": True,
            }
        },
    )

    resolved = config["resolved"]

    assert resolved["ev_bus_ids"] == []
    assert resolved["battery_bus_ids"] == []
    assert resolved["pv_bus_ids"] == ["h0"]


def test_manual_override_wins_over_share_percent_selection():
    household_ids = ["h0", "h1"]

    config = _build(
        _network(),
        household_ids,
        ev_share=0.0,
        overrides={
            "h0": {
                "has_ev": True,
            }
        },
    )

    assert config["resolved"]["ev_bus_ids"] == ["h0"]


def test_manual_battery_and_pv_overrides_work_without_real_device_data():
    household_ids = ["h0", "h1"]

    config = _build(
        _network(),
        household_ids,
        ev_share=0.0,
        hp_share=0.0,
        overrides={
            "h0": {
                "has_battery": True,
                "has_pv": True,
            }
        },
    )

    resolved = config["resolved"]

    assert resolved["battery_bus_ids"] == ["h0"]
    assert resolved["pv_bus_ids"] == ["h0"]


def test_individual_load_scaling_override_wins_over_global_value():
    household_ids = ["h0", "h1"]

    config = _build(
        _network(),
        household_ids,
        load_scaling=1.0,
        overrides={
            "h0": {
                "load_scaling_factor": 1.5,
            }
        },
    )

    resolved = config["resolved"]

    assert resolved["load_scaling_by_bus"]["h0"] == 1.5
    assert resolved["load_scaling_by_bus"]["h1"] == 1.0


def test_global_load_scaling_is_applied_to_all_households_without_override():
    household_ids = ["h0", "h1", "h2"]

    config = _build(
        _network(),
        household_ids,
        load_scaling=1.2,
    )

    resolved = config["resolved"]

    assert resolved["load_scaling_by_bus"] == {
        "h0": 1.2,
        "h1": 1.2,
        "h2": 1.2,
    }