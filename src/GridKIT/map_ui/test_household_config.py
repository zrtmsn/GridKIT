from types import SimpleNamespace

from core.models import HouseholdDevices
from map_ui.household_config import build_household_configuration


# Shared bounds fixture values.
BOUNDS_SOUTH = 49.0
BOUNDS_WEST = 8.4
BOUNDS_NORTH = 49.01
BOUNDS_EAST = 8.41
BOUNDS_AREA_KM2 = 1.0

# Shared test network metadata.
TEST_NETWORK_ID = "n"
TEST_AREA_NAME = "test-area"

# Default scenario values used by the test helper.
DEFAULT_EV_SHARE_PERCENT = 30.0
DEFAULT_HEAT_PUMP_SHARE_PERCENT = 20.0
DEFAULT_BATTERY_SHARE_PERCENT = 0.0
DEFAULT_PV_SHARE_PERCENT = 0.0
DEFAULT_LOAD_SCALING_FACTOR = 1.0
DEFAULT_SELECTION_SEED = 42

# Test values for deterministic share and scaling checks.
HOUSEHOLD_COUNT_FOR_SHARE_TEST = 10
EV_SHARE_HALF_PERCENT = 50.0
ZERO_SHARE_PERCENT = 0.0
EXPECTED_HALF_SHARE_COUNT = 5
CUSTOM_LOAD_SCALING_FACTOR = 1.5
GLOBAL_LOAD_SCALING_FACTOR = 1.2

# Test device capacity values.
OBJECT_PV_KWP = 6.5
OBJECT_BATTERY_KWH = 12.0
DICT_PV_KWP = 7.5
DICT_BATTERY_KWH = 10.0


class _Bounds:
    """Minimal bounds object required by build_household_configuration."""

    def model_dump(self):
        return {
            "south": BOUNDS_SOUTH,
            "west": BOUNDS_WEST,
            "north": BOUNDS_NORTH,
            "east": BOUNDS_EAST,
        }

    def approx_area_km2(self):
        return BOUNDS_AREA_KM2


def _network(
    household_devices=None,
    household_load_profile_kw=None,
    ev_availability=None,
):
    """Create a minimal network object for household configuration tests."""
    return SimpleNamespace(
        network_id=TEST_NETWORK_ID,
        area_name=TEST_AREA_NAME,
        household_devices=household_devices or {},
        household_load_profile_kw=household_load_profile_kw or {},
        ev_availability=ev_availability or {},
    )


def _build(
    network,
    household_ids,
    overrides=None,
    ev_share=DEFAULT_EV_SHARE_PERCENT,
    hp_share=DEFAULT_HEAT_PUMP_SHARE_PERCENT,
    battery_share=DEFAULT_BATTERY_SHARE_PERCENT,
    pv_share=DEFAULT_PV_SHARE_PERCENT,
    load_scaling=DEFAULT_LOAD_SCALING_FACTOR,
    seed=DEFAULT_SELECTION_SEED,
):
    """Build a household configuration with compact test defaults."""
    return build_household_configuration(
        network=network,
        selected_bounds=_Bounds(),
        household_ids=household_ids,
        ev_share_percent=ev_share,
        heat_pump_share_percent=hp_share,
        global_load_scaling_factor=load_scaling,
        selection_seed=seed,
        household_overrides=overrides or {},
        battery_share_percent=battery_share,
        pv_share_percent=pv_share,
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
            pv_kwp=OBJECT_PV_KWP,
            battery_kwh=OBJECT_BATTERY_KWH,
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
    assert resolved["pv_kwp_by_bus"] == {"h0": OBJECT_PV_KWP}
    assert resolved["battery_kwh_by_bus"] == {"h0": OBJECT_BATTERY_KWH}
    assert config["scenario_assumptions"]["uses_gridcreator_defaults"] is True


def test_supports_dictionary_device_data():
    household_ids = ["h0"]

    devices = {
        "h0": {
            "ev": True,
            "heat_pump": True,
            "battery": True,
            "pv": True,
            "pv_kwp": DICT_PV_KWP,
            "battery_kwh": DICT_BATTERY_KWH,
        }
    }

    config = _build(_network(household_devices=devices), household_ids)

    resolved = config["resolved"]

    assert resolved["ev_bus_ids"] == ["h0"]
    assert resolved["heat_pump_bus_ids"] == ["h0"]
    assert resolved["battery_bus_ids"] == ["h0"]
    assert resolved["pv_bus_ids"] == ["h0"]
    assert resolved["pv_kwp_by_bus"] == {"h0": DICT_PV_KWP}
    assert resolved["battery_kwh_by_bus"] == {"h0": DICT_BATTERY_KWH}


def test_uses_share_percent_when_no_real_device_data_exists():
    household_ids = [f"h{i}" for i in range(HOUSEHOLD_COUNT_FOR_SHARE_TEST)]

    config = _build(
        _network(),
        household_ids,
        ev_share=EV_SHARE_HALF_PERCENT,
        hp_share=ZERO_SHARE_PERCENT,
    )

    resolved = config["resolved"]

    assert len(resolved["ev_bus_ids"]) == EXPECTED_HALF_SHARE_COUNT
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
        ev_share=ZERO_SHARE_PERCENT,
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
        ev_share=ZERO_SHARE_PERCENT,
        hp_share=ZERO_SHARE_PERCENT,
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
        load_scaling=DEFAULT_LOAD_SCALING_FACTOR,
        overrides={
            "h0": {
                "load_scaling_factor": CUSTOM_LOAD_SCALING_FACTOR,
            }
        },
    )

    resolved = config["resolved"]

    assert resolved["load_scaling_by_bus"]["h0"] == CUSTOM_LOAD_SCALING_FACTOR
    assert resolved["load_scaling_by_bus"]["h1"] == DEFAULT_LOAD_SCALING_FACTOR


def test_global_load_scaling_is_applied_to_all_households_without_override():
    household_ids = ["h0", "h1", "h2"]

    config = _build(
        _network(),
        household_ids,
        load_scaling=GLOBAL_LOAD_SCALING_FACTOR,
    )

    resolved = config["resolved"]

    assert resolved["load_scaling_by_bus"] == {
        "h0": GLOBAL_LOAD_SCALING_FACTOR,
        "h1": GLOBAL_LOAD_SCALING_FACTOR,
        "h2": GLOBAL_LOAD_SCALING_FACTOR,
    }