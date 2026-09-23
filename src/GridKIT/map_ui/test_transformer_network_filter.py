from types import SimpleNamespace

from map_ui.transformer_network_filter import (
    filter_network_by_transformer,
    reachable_bus_ids_for_transformer,
    selectable_transformer_ids,
    transformer_group_members,
)


TEST_NETWORK_ID = "test_network"
TEST_AREA_NAME = "test_area"

DEFAULT_BUS_X_COORD = 8.0
DEFAULT_BUS_Y_COORD = 49.0
DEFAULT_BUS_V_NOM_KV = 0.4

DEFAULT_LINE_LENGTH_KM = 0.01
DEFAULT_LINE_R_OHM_PER_KM = 0.1
DEFAULT_LINE_X_OHM_PER_KM = 0.01
DEFAULT_LINE_MAX_I_KA = 0.2

DEFAULT_TRAFO_S_NOM_MVA = 0.4
DEFAULT_TRAFO_VN_HV_KV = 20.0
DEFAULT_TRAFO_VN_LV_KV = 0.4

TRAFO_1_ID = "transformer_1"
TRAFO_2_ID = "transformer_2"
TRAFO_3_ID = "transformer_3"

TRAFO_1_HV_BUS = "transformer_1_hv"
TRAFO_1_LV_BUS = "transformer_1_lv"
TRAFO_2_HV_BUS = "transformer_2_hv"
TRAFO_2_LV_BUS = "transformer_2_lv"
TRAFO_3_HV_BUS = "transformer_3_hv"
TRAFO_3_LV_BUS = "transformer_3_lv"

HOUSEHOLD_1_ID = "household_1"
HOUSEHOLD_2_ID = "household_2"

LINE_1_ID = "line_1"
LINE_2_ID = "line_2"
LINE_3_ID = "line_3"

LOAD_PROFILE_A = [1.0, 2.0]
LOAD_PROFILE_B = [3.0, 4.0]
EV_AVAILABILITY_A = [1, 1]
EV_AVAILABILITY_B = [0, 0]


class FakeNetwork(SimpleNamespace):
    """Minimal network object that mimics Pydantic's model_copy behavior."""

    def model_copy(self, deep=False, update=None):
        data = dict(self.__dict__)
        data.update(update or {})
        return FakeNetwork(**data)


def _bus(bus_id, x=DEFAULT_BUS_X_COORD, y=DEFAULT_BUS_Y_COORD):
    """Create a minimal bus object for transformer-filter tests."""
    return SimpleNamespace(
        bus_id=bus_id,
        x_coord=x,
        y_coord=y,
        v_nom_kv=DEFAULT_BUS_V_NOM_KV,
    )


def _line(line_id, from_bus, to_bus):
    """Create a minimal line object connecting two buses."""
    return SimpleNamespace(
        line_id=line_id,
        from_bus=from_bus,
        to_bus=to_bus,
        length_km=DEFAULT_LINE_LENGTH_KM,
        r_ohm_per_km=DEFAULT_LINE_R_OHM_PER_KM,
        x_ohm_per_km=DEFAULT_LINE_X_OHM_PER_KM,
        max_i_ka=DEFAULT_LINE_MAX_I_KA,
    )


def _trafo(trafo_id, hv_bus, lv_bus):
    """Create a minimal transformer object."""
    return SimpleNamespace(
        trafo_id=trafo_id,
        hv_bus=hv_bus,
        lv_bus=lv_bus,
        s_nom_mva=DEFAULT_TRAFO_S_NOM_MVA,
        vn_hv_kv=DEFAULT_TRAFO_VN_HV_KV,
        vn_lv_kv=DEFAULT_TRAFO_VN_LV_KV,
    )


def _network(
    transformers,
    buses,
    lines=None,
    household_ids=None,
    household_devices=None,
    household_load_profile_kw=None,
    ev_availability=None,
):
    """Create a minimal network object used by the filter helpers."""
    return FakeNetwork(
        network_id=TEST_NETWORK_ID,
        area_name=TEST_AREA_NAME,
        transformers=list(transformers),
        buses=list(buses),
        lines=list(lines or []),
        household_bus_ids=list(household_ids or []),
        household_devices=household_devices or {},
        household_load_profile_kw=household_load_profile_kw or {},
        ev_availability=ev_availability or {},
    )


# Transformer areas are grouped structurally by transformer.lv_bus.
# This matches grid_model.builder.assign_clean_ids for reinforced transformers.
def test_selectable_transformer_ids_show_one_area_per_lv_bus():
    regular_1 = _trafo(TRAFO_1_ID, TRAFO_1_HV_BUS, TRAFO_1_LV_BUS)
    reinforced_2 = _trafo(TRAFO_2_ID, TRAFO_1_HV_BUS, TRAFO_1_LV_BUS)
    other_area = _trafo(TRAFO_3_ID, TRAFO_3_HV_BUS, TRAFO_3_LV_BUS)

    net = _network(transformers=[regular_1, reinforced_2, other_area], buses=[])

    assert selectable_transformer_ids(net) == [TRAFO_1_ID, TRAFO_3_ID]


def test_transformer_group_members_share_the_same_lv_bus():
    regular_1 = _trafo(TRAFO_1_ID, TRAFO_1_HV_BUS, TRAFO_1_LV_BUS)
    reinforced_2 = _trafo(TRAFO_2_ID, TRAFO_1_HV_BUS, TRAFO_1_LV_BUS)
    other_area = _trafo(TRAFO_3_ID, TRAFO_3_HV_BUS, TRAFO_3_LV_BUS)

    net = _network(transformers=[regular_1, reinforced_2, other_area], buses=[])

    members = transformer_group_members(net, TRAFO_1_ID)

    assert {str(member.trafo_id) for member in members} == {TRAFO_1_ID, TRAFO_2_ID}


def test_reachable_bus_ids_follow_topology_only():
    # A household is included only when a line connects it to the transformer.
    trafo = _trafo(TRAFO_1_ID, TRAFO_1_HV_BUS, TRAFO_1_LV_BUS)
    connected_household = HOUSEHOLD_1_ID
    disconnected_household = HOUSEHOLD_2_ID

    net = _network(
        transformers=[trafo],
        buses=[
            _bus(TRAFO_1_HV_BUS),
            _bus(TRAFO_1_LV_BUS),
            _bus(connected_household),
            _bus(disconnected_household),
        ],
        lines=[_line(LINE_1_ID, TRAFO_1_LV_BUS, connected_household)],
        household_ids=[connected_household, disconnected_household],
    )

    reachable = reachable_bus_ids_for_transformer(net, TRAFO_1_ID)

    assert TRAFO_1_HV_BUS in reachable
    assert TRAFO_1_LV_BUS in reachable
    assert connected_household in reachable
    assert disconnected_household not in reachable


def test_reachable_bus_ids_stop_at_another_transformers_buses():
    # The search may follow feeder lines, but it must not cross into another
    # transformer's busbar and continue to that transformer's households.
    trafo_a = _trafo(TRAFO_1_ID, TRAFO_1_HV_BUS, TRAFO_1_LV_BUS)
    trafo_b = _trafo(TRAFO_2_ID, TRAFO_2_HV_BUS, TRAFO_2_LV_BUS)
    household_a = HOUSEHOLD_1_ID
    household_b = HOUSEHOLD_2_ID

    net = _network(
        transformers=[trafo_a, trafo_b],
        buses=[
            _bus(TRAFO_1_HV_BUS),
            _bus(TRAFO_1_LV_BUS),
            _bus(household_a),
            _bus(TRAFO_2_HV_BUS),
            _bus(TRAFO_2_LV_BUS),
            _bus(household_b),
        ],
        lines=[
            _line(LINE_1_ID, TRAFO_1_LV_BUS, household_a),
            _line(LINE_2_ID, household_a, TRAFO_2_LV_BUS),
            _line(LINE_3_ID, TRAFO_2_LV_BUS, household_b),
        ],
        household_ids=[household_a, household_b],
    )

    reachable_a = reachable_bus_ids_for_transformer(net, TRAFO_1_ID)

    assert TRAFO_1_HV_BUS in reachable_a
    assert TRAFO_1_LV_BUS in reachable_a
    assert household_a in reachable_a
    assert TRAFO_2_LV_BUS not in reachable_a
    assert household_b not in reachable_a


def test_filter_network_by_transformer_keeps_only_its_own_feeder():
    trafo_a = _trafo(TRAFO_1_ID, TRAFO_1_HV_BUS, TRAFO_1_LV_BUS)
    trafo_b = _trafo(TRAFO_2_ID, TRAFO_2_HV_BUS, TRAFO_2_LV_BUS)
    household_a = HOUSEHOLD_1_ID
    household_b = HOUSEHOLD_2_ID

    net = _network(
        transformers=[trafo_a, trafo_b],
        buses=[
            _bus(TRAFO_1_HV_BUS),
            _bus(TRAFO_1_LV_BUS),
            _bus(household_a),
            _bus(TRAFO_2_HV_BUS),
            _bus(TRAFO_2_LV_BUS),
            _bus(household_b),
        ],
        lines=[
            _line(LINE_1_ID, TRAFO_1_LV_BUS, household_a),
            _line(LINE_2_ID, TRAFO_2_LV_BUS, household_b),
        ],
        household_ids=[household_a, household_b],
        household_devices={household_a: {"ev": True}, household_b: {"ev": True}},
        household_load_profile_kw={
            household_a: LOAD_PROFILE_A,
            household_b: LOAD_PROFILE_B,
        },
        ev_availability={
            household_a: EV_AVAILABILITY_A,
            household_b: EV_AVAILABILITY_B,
        },
    )

    filtered = filter_network_by_transformer(net, TRAFO_1_ID)

    assert filtered.household_bus_ids == [household_a]
    assert {str(bus.bus_id) for bus in filtered.buses} == {
        TRAFO_1_HV_BUS,
        TRAFO_1_LV_BUS,
        household_a,
    }
    assert filtered.household_devices == {household_a: {"ev": True}}
    assert filtered.household_load_profile_kw == {household_a: LOAD_PROFILE_A}
    assert filtered.ev_availability == {household_a: EV_AVAILABILITY_A}
    assert len(filtered.transformers) == 1
    assert filtered.transformers[0].trafo_id == TRAFO_1_ID


def test_filter_network_by_transformer_includes_reinforced_partner():
    regular_1 = _trafo(TRAFO_1_ID, TRAFO_1_HV_BUS, TRAFO_1_LV_BUS)
    reinforced_2 = _trafo(TRAFO_2_ID, TRAFO_1_HV_BUS, TRAFO_1_LV_BUS)
    household_a = HOUSEHOLD_1_ID

    net = _network(
        transformers=[regular_1, reinforced_2],
        buses=[
            _bus(TRAFO_1_HV_BUS),
            _bus(TRAFO_1_LV_BUS),
            _bus(household_a),
        ],
        lines=[_line(LINE_1_ID, TRAFO_1_LV_BUS, household_a)],
        household_ids=[household_a],
    )

    filtered = filter_network_by_transformer(net, TRAFO_1_ID)

    assert {str(transformer.trafo_id) for transformer in filtered.transformers} == {
        TRAFO_1_ID,
        TRAFO_2_ID,
    }