# grid_model/test_builder.py
import pandas as pd
import pypsa
import pytest

import core.constants as const
from .builder import OSMNetworkBuilder, StubNetworkBuilder, assign_clean_ids
from core.models import BusModel, GridNetwork, LineModel, TransformerModel

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

    # ids are reassigned to clean, human-readable ones (see assign_clean_ids)
    # — "fully_equipped" sorts before "no_devices", so it becomes household_1.
    assert result.household_bus_ids == ["household_1", "household_2"]

    equipped = result.household_devices["household_1"]
    assert equipped.ev is True and equipped.heat_pump is True
    assert equipped.pv is True and equipped.pv_kwp == 6.0
    assert equipped.battery is True and equipped.battery_kwh == 5.0

    empty = result.household_devices["household_2"]
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

    dev = result.household_devices["household_1"]
    assert dev.pv_kwp == const.PV_PEAK_KWP_MAX
    assert dev.battery_kwh == const.BATTERY_CAPACITY_KWH_MAX


# ══════════════════════════════════════════════════════════════
# assign_clean_ids — GridCreator's/ding0's verbose ids get replaced with
# short, human-readable ones (household_N / transformer_N_hv|lv / bus_N /
# line_N), consistently across buses, lines, transformers, and every dict
# keyed by bus_id.
# ══════════════════════════════════════════════════════════════
def test_assign_clean_ids_renames_everything_consistently():
    buses = [
        BusModel(bus_id="BusBar_mv"), BusModel(bus_id="BusBar_lv"),
        BusModel(bus_id="BranchTee_1"), BusModel(bus_id="BranchTee_building_99"),
    ]
    lines = [
        LineModel(line_id="Branch_z_line", from_bus="BusBar_lv", to_bus="BranchTee_1",
                   length_km=0.1, r_ohm_per_km=0.3, x_ohm_per_km=0.08, max_i_ka=0.2),
        LineModel(line_id="Branch_a_line", from_bus="BranchTee_1", to_bus="BranchTee_building_99",
                   length_km=0.1, r_ohm_per_km=0.3, x_ohm_per_km=0.08, max_i_ka=0.2),
    ]
    transformers = [
        TransformerModel(trafo_id="Transformer_weird_name", hv_bus="BusBar_mv", lv_bus="BusBar_lv", s_nom_mva=0.4),
    ]
    household_bus_ids = ["BranchTee_building_99"]

    (new_buses, new_lines, new_transformers, new_household_bus_ids,
     _, _, _) = assign_clean_ids(buses, lines, transformers, household_bus_ids)

    assert {b.bus_id for b in new_buses} == {"transformer_1_hv", "transformer_1_lv", "household_1", "bus_1"}
    assert new_transformers[0].trafo_id == "transformer_1"
    assert new_transformers[0].hv_bus == "transformer_1_hv"
    assert new_transformers[0].lv_bus == "transformer_1_lv"
    assert new_household_bus_ids == ["household_1"]
    assert {ln.line_id for ln in new_lines} == {"line_1", "line_2"}
    # from_bus/to_bus were translated through the same map, not left raw
    clean_ids = {"transformer_1_hv", "transformer_1_lv", "household_1", "bus_1"}
    assert all(ln.from_bus in clean_ids and ln.to_bus in clean_ids for ln in new_lines)


def test_assign_clean_ids_shares_busbar_name_across_parallel_transformers():
    # a real ding0 "reinforced" pair: two transformers on the identical lv_bus
    buses = [BusModel(bus_id="mv"), BusModel(bus_id="lv")]
    transformers = [
        TransformerModel(trafo_id="t_a", hv_bus="mv", lv_bus="lv", s_nom_mva=0.4),
        TransformerModel(trafo_id="t_b", hv_bus="mv", lv_bus="lv", s_nom_mva=0.4),
    ]

    _, _, new_transformers, _, _, _, _ = assign_clean_ids(buses, [], transformers, [])

    assert new_transformers[0].trafo_id == "transformer_1"
    assert new_transformers[1].trafo_id == "transformer_2"
    # both point at the SAME renamed busbar — it's physically one bus
    assert new_transformers[0].lv_bus == new_transformers[1].lv_bus == "transformer_1_lv"


def test_assign_clean_ids_is_deterministic():
    # numbering is alphabetical over the ORIGINAL ids ("a" < "z"), independent
    # of the order they're passed in; the returned list still mirrors the
    # input's order, just with each id translated.
    buses = [BusModel(bus_id="z"), BusModel(bus_id="a")]
    first = assign_clean_ids(buses, [], [], ["z", "a"])
    second = assign_clean_ids(buses, [], [], ["z", "a"])
    assert first[3] == second[3] == ["household_2", "household_1"]