# grid_model/test_builder.py
import pandas as pd
import pypsa
import pytest

import core.constants as const
from .builder import OSMNetworkBuilder, StubNetworkBuilder
from core.models import GridNetwork

@pytest.fixture
def network():
    return StubNetworkBuilder().build()

def test_stub_builder_returns_grid_network(network):
    assert isinstance(network, GridNetwork)

def test_stub_network_has_households(network):
    assert network.n_households > 0

def test_stub_network_has_buses(network):
    assert len(network.buses) > 0

def test_stub_network_has_lines(network):
    assert len(network.lines) > 0

def test_stub_network_has_transformer(network):
    assert len(network.transformers) > 0


# ══════════════════════════════════════════════════════════════
# OSMNetworkBuilder._to_grid_network — real GridCreator device assignment
# (Power_solar/storage/Power_E_car/Power_HP columns in buses_df, written by
# gcp_assignment/gcp_fill). Pure method, no conda/GridCreator subprocess
# needed — fabricate a minimal pypsa.Network + buses_df directly.
# ══════════════════════════════════════════════════════════════
def _fake_grid(bus_ids: list[str]) -> pypsa.Network:
    grid = pypsa.Network()
    for bus_id in bus_ids:
        grid.add("Bus", bus_id, v_nom=0.4, x=0.0, y=0.0)
    return grid


def test_to_grid_network_reads_gridcreator_device_assignment():
    grid = _fake_grid(["fully_equipped", "no_devices"])
    buses_df = pd.DataFrame(
        index=["fully_equipped", "no_devices"],
        data={
            "Haushalte": [2, 1],
            "Power_solar": [6.0, 0.0],
            "storage": [5.0, 0.0],
            "Power_E_car": [3.0, 0.0],
            "Power_HP": [1.5, 0.0],
        },
    )
    builder = OSMNetworkBuilder(top=0, bottom=0, left=0, right=0, scenario="test")

    result = builder._to_grid_network(grid, buses_df)

    equipped = result.household_devices["fully_equipped"]
    assert equipped.ev is True and equipped.heat_pump is True
    assert equipped.pv is True and equipped.pv_kwp == 6.0
    assert equipped.battery is True and equipped.battery_kwh == 5.0

    empty = result.household_devices["no_devices"]
    assert not (empty.ev or empty.heat_pump or empty.pv or empty.battery)
    assert empty.pv_kwp is None and empty.battery_kwh is None


def test_to_grid_network_clamps_out_of_range_capacities():
    grid = _fake_grid(["oversized"])
    buses_df = pd.DataFrame(
        index=["oversized"],
        data={"Haushalte": [1], "Power_solar": [50.0], "storage": [99.0],
              "Power_E_car": [0.0], "Power_HP": [0.0]},
    )
    builder = OSMNetworkBuilder(top=0, bottom=0, left=0, right=0, scenario="test")

    result = builder._to_grid_network(grid, buses_df)

    dev = result.household_devices["oversized"]
    assert dev.pv_kwp == const.PV_PEAK_KWP_MAX
    assert dev.battery_kwh == const.BATTERY_CAPACITY_KWH_MAX