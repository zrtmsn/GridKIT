from types import SimpleNamespace

from map_ui.transformer_network_filter import (
    filter_network_by_transformer,
    reachable_bus_ids_for_transformer,
    selectable_transformer_ids,
    transformer_group_members,
)


class FakeNetwork(SimpleNamespace):
    def model_copy(self, deep=False, update=None):
        data = dict(self.__dict__)
        data.update(update or {})
        return FakeNetwork(**data)


def _bus(bus_id, x=8.0, y=49.0):
    return SimpleNamespace(bus_id=bus_id, x_coord=x, y_coord=y, v_nom_kv=0.4)


def _line(line_id, from_bus, to_bus):
    return SimpleNamespace(
        line_id=line_id, from_bus=from_bus, to_bus=to_bus,
        length_km=0.01, r_ohm_per_km=0.1, x_ohm_per_km=0.01, max_i_ka=0.2,
    )


def _trafo(trafo_id, hv_bus, lv_bus):
    return SimpleNamespace(
        trafo_id=trafo_id, hv_bus=hv_bus, lv_bus=lv_bus,
        s_nom_mva=0.4, vn_hv_kv=20.0, vn_lv_kv=0.4,
    )


def _network(
    transformers, buses, lines=None, household_ids=None,
    household_devices=None, household_load_profile_kw=None, ev_availability=None,
):
    return FakeNetwork(
        network_id="test_network",
        area_name="test_area",
        transformers=list(transformers),
        buses=list(buses),
        lines=list(lines or []),
        household_bus_ids=list(household_ids or []),
        household_devices=household_devices or {},
        household_load_profile_kw=household_load_profile_kw or {},
        ev_availability=ev_availability or {},
    )


# Grouping is now purely structural (transformer.lv_bus), not derived from
# parsing trafo_id/bus_id strings — see grid_model.builder.assign_clean_ids,
# which already guarantees a reinforced pair shares one lv_bus.

def test_selectable_transformer_ids_show_one_area_per_lv_bus():
    regular_1 = _trafo("transformer_1", "transformer_1_hv", "transformer_1_lv")
    reinforced_2 = _trafo("transformer_2", "transformer_1_hv", "transformer_1_lv")
    other_area = _trafo("transformer_3", "transformer_3_hv", "transformer_3_lv")

    net = _network(transformers=[regular_1, reinforced_2, other_area], buses=[])

    assert selectable_transformer_ids(net) == ["transformer_1", "transformer_3"]


def test_transformer_group_members_share_the_same_lv_bus():
    regular_1 = _trafo("transformer_1", "transformer_1_hv", "transformer_1_lv")
    reinforced_2 = _trafo("transformer_2", "transformer_1_hv", "transformer_1_lv")
    other_area = _trafo("transformer_3", "transformer_3_hv", "transformer_3_lv")

    net = _network(transformers=[regular_1, reinforced_2, other_area], buses=[])

    members = transformer_group_members(net, "transformer_1")

    assert {str(m.trafo_id) for m in members} == {"transformer_1", "transformer_2"}


def test_reachable_bus_ids_follow_topology_only():
    # connected via a line -> included; no connecting line at all -> excluded,
    # regardless of id/naming
    trafo = _trafo("transformer_1", "transformer_1_hv", "transformer_1_lv")
    connected_household = "household_1"
    disconnected_household = "household_2"

    net = _network(
        transformers=[trafo],
        buses=[
            _bus("transformer_1_hv"), _bus("transformer_1_lv"),
            _bus(connected_household), _bus(disconnected_household),
        ],
        lines=[_line("line_1", "transformer_1_lv", connected_household)],
        household_ids=[connected_household, disconnected_household],
    )

    reachable = reachable_bus_ids_for_transformer(net, "transformer_1")

    assert "transformer_1_hv" in reachable
    assert "transformer_1_lv" in reachable
    assert connected_household in reachable
    assert disconnected_household not in reachable


def test_reachable_bus_ids_stop_at_another_transformers_buses():
    trafo_a = _trafo("transformer_1", "transformer_1_hv", "transformer_1_lv")
    trafo_b = _trafo("transformer_2", "transformer_2_hv", "transformer_2_lv")
    household_a = "household_1"
    household_b = "household_2"

    net = _network(
        transformers=[trafo_a, trafo_b],
        buses=[
            _bus("transformer_1_hv"), _bus("transformer_1_lv"), _bus(household_a),
            _bus("transformer_2_hv"), _bus("transformer_2_lv"), _bus(household_b),
        ],
        lines=[
            _line("line_1", "transformer_1_lv", household_a),
            _line("line_2", "transformer_2_lv", household_b),
        ],
        household_ids=[household_a, household_b],
    )

    reachable_a = reachable_bus_ids_for_transformer(net, "transformer_1")

    assert household_a in reachable_a
    assert household_b not in reachable_a


def test_filter_network_by_transformer_keeps_only_its_own_feeder():
    trafo_a = _trafo("transformer_1", "transformer_1_hv", "transformer_1_lv")
    trafo_b = _trafo("transformer_2", "transformer_2_hv", "transformer_2_lv")
    household_a = "household_1"
    household_b = "household_2"

    net = _network(
        transformers=[trafo_a, trafo_b],
        buses=[
            _bus("transformer_1_hv"), _bus("transformer_1_lv"), _bus(household_a),
            _bus("transformer_2_hv"), _bus("transformer_2_lv"), _bus(household_b),
        ],
        lines=[
            _line("line_1", "transformer_1_lv", household_a),
            _line("line_2", "transformer_2_lv", household_b),
        ],
        household_ids=[household_a, household_b],
        household_devices={household_a: {"ev": True}, household_b: {"ev": True}},
        household_load_profile_kw={household_a: [1.0, 2.0], household_b: [3.0, 4.0]},
        ev_availability={household_a: [1, 1], household_b: [0, 0]},
    )

    filtered = filter_network_by_transformer(net, "transformer_1")

    assert filtered.household_bus_ids == [household_a]
    assert {str(bus.bus_id) for bus in filtered.buses} == {
        "transformer_1_hv", "transformer_1_lv", household_a,
    }
    assert filtered.household_devices == {household_a: {"ev": True}}
    assert filtered.household_load_profile_kw == {household_a: [1.0, 2.0]}
    assert filtered.ev_availability == {household_a: [1, 1]}
    assert len(filtered.transformers) == 1
    assert filtered.transformers[0].trafo_id == "transformer_1"


def test_filter_network_by_transformer_includes_reinforced_partner():
    regular_1 = _trafo("transformer_1", "transformer_1_hv", "transformer_1_lv")
    reinforced_2 = _trafo("transformer_2", "transformer_1_hv", "transformer_1_lv")
    household_a = "household_1"

    net = _network(
        transformers=[regular_1, reinforced_2],
        buses=[_bus("transformer_1_hv"), _bus("transformer_1_lv"), _bus(household_a)],
        lines=[_line("line_1", "transformer_1_lv", household_a)],
        household_ids=[household_a],
    )

    filtered = filter_network_by_transformer(net, "transformer_1")

    assert {str(t.trafo_id) for t in filtered.transformers} == {"transformer_1", "transformer_2"}
