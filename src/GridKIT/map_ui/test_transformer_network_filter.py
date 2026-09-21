from types import SimpleNamespace

from map_ui.transformer_network_filter import (
    bus_ids_for_lv_grid,
    filter_network_by_transformer,
    household_ids_for_lv_grid,
    lv_grid_id_from_bus_id,
    parse_transformer_id,
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
    return SimpleNamespace(
        bus_id=bus_id,
        x_coord=x,
        y_coord=y,
        v_nom_kv=0.4,
    )


def _line(line_id, from_bus, to_bus):
    return SimpleNamespace(
        line_id=line_id,
        from_bus=from_bus,
        to_bus=to_bus,
        length_km=0.01,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.01,
        max_i_ka=0.2,
    )


def _trafo(trafo_id, hv_bus="hv_bus", lv_bus="lv_bus"):
    return SimpleNamespace(
        trafo_id=trafo_id,
        hv_bus=hv_bus,
        lv_bus=lv_bus,
        s_nom_mva=0.4,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
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


def test_parse_transformer_id_regular_transformer():
    info = parse_transformer_id("Transformer_lv_grid_5690200024_1")

    assert info["lv_grid_id"] == "5690200024"
    assert info["trafo_number"] == "1"
    assert info["is_reinforced"] is False
    assert info["role"] == "regular_transformer"


def test_parse_transformer_id_reinforced_transformer():
    info = parse_transformer_id("Transformer_lv_grid_5690200024_reinforced_2")

    assert info["lv_grid_id"] == "5690200024"
    assert info["trafo_number"] == "2"
    assert info["is_reinforced"] is True
    assert info["role"] == "reinforced_transformer"


def test_lv_grid_id_from_household_bus_id():
    bus_id = "BranchTee_mvgd_35996_lvgd_5690200024_building_1556697"

    assert lv_grid_id_from_bus_id(bus_id) == "5690200024"


def test_lv_grid_id_from_bus_id_returns_none_when_missing():
    assert lv_grid_id_from_bus_id("some_bus_without_lvgd_information") is None


def test_household_ids_for_lv_grid_returns_only_matching_households():
    household_a = "BranchTee_mvgd_35996_lvgd_5690200024_building_1556697"
    household_b = "BranchTee_mvgd_35996_lvgd_5690200099_building_1556698"

    net = _network(
        transformers=[],
        buses=[
            _bus(household_a),
            _bus(household_b),
        ],
        household_ids=[
            household_a,
            household_b,
        ],
    )

    assert household_ids_for_lv_grid(net, "5690200024") == {household_a}


def test_bus_ids_for_lv_grid_returns_only_matching_buses():
    bus_a = "BranchTee_mvgd_35996_lvgd_5690200024_building_1556697"
    bus_b = "BranchTee_mvgd_35996_lvgd_5690200099_building_1556698"
    bus_without_lvgd = "ordinary_bus"

    net = _network(
        transformers=[],
        buses=[
            _bus(bus_a),
            _bus(bus_b),
            _bus(bus_without_lvgd),
        ],
    )

    assert bus_ids_for_lv_grid(net, "5690200024") == {bus_a}


def test_selectable_transformer_ids_show_one_area_per_lv_grid():
    regular_1 = _trafo("Transformer_lv_grid_5690200024_1")
    regular_2_same_grid = _trafo("Transformer_lv_grid_5690200024_2")
    reinforced_same_grid = _trafo("Transformer_lv_grid_5690200024_reinforced_3")
    other_grid = _trafo("Transformer_lv_grid_5690200099_1")

    net = _network(
        transformers=[
            regular_1,
            regular_2_same_grid,
            reinforced_same_grid,
            other_grid,
        ],
        buses=[],
    )

    assert selectable_transformer_ids(net) == [
        "Transformer_lv_grid_5690200024_1",
        "Transformer_lv_grid_5690200099_1",
    ]


def test_transformer_group_members_include_same_lv_grid_transformers():
    regular_1 = _trafo("Transformer_lv_grid_5690200024_1")
    regular_2_same_grid = _trafo("Transformer_lv_grid_5690200024_2")
    reinforced_same_grid = _trafo("Transformer_lv_grid_5690200024_reinforced_3")
    other_grid = _trafo("Transformer_lv_grid_5690200099_1")

    net = _network(
        transformers=[
            regular_1,
            regular_2_same_grid,
            reinforced_same_grid,
            other_grid,
        ],
        buses=[],
    )

    members = transformer_group_members(net, "Transformer_lv_grid_5690200024_1")
    member_ids = {str(member.trafo_id) for member in members}

    assert member_ids == {
        "Transformer_lv_grid_5690200024_1",
        "Transformer_lv_grid_5690200024_2",
        "Transformer_lv_grid_5690200024_reinforced_3",
    }


def test_reachable_bus_ids_include_household_with_same_lv_grid_even_without_line_connection():
    trafo = _trafo(
        "Transformer_lv_grid_5690200024_1",
        hv_bus="hv_bus",
        lv_bus="lv_bus",
    )

    matching_household = "BranchTee_mvgd_35996_lvgd_5690200024_building_1556936"
    other_household = "BranchTee_mvgd_35996_lvgd_5690200099_building_1556937"

    net = _network(
        transformers=[trafo],
        buses=[
            _bus("hv_bus"),
            _bus("lv_bus"),
            _bus(matching_household),
            _bus(other_household),
        ],
        lines=[],
        household_ids=[
            matching_household,
            other_household,
        ],
    )

    reachable = reachable_bus_ids_for_transformer(
        net,
        "Transformer_lv_grid_5690200024_1",
    )

    assert "hv_bus" in reachable
    assert "lv_bus" in reachable
    assert matching_household in reachable
    assert other_household not in reachable


def test_filter_network_by_transformer_keeps_only_matching_lv_grid_households_and_related_data():
    trafo = _trafo(
        "Transformer_lv_grid_5690200024_1",
        hv_bus="hv_bus",
        lv_bus="lv_bus",
    )

    matching_household = "BranchTee_mvgd_35996_lvgd_5690200024_building_1556936"
    non_matching_household = "BranchTee_mvgd_35996_lvgd_5690200099_building_1556937"

    net = _network(
        transformers=[trafo],
        buses=[
            _bus("hv_bus"),
            _bus("lv_bus"),
            _bus(matching_household),
            _bus(non_matching_household),
        ],
        lines=[
            _line("line_1", "lv_bus", matching_household),
            _line("line_2", "lv_bus", non_matching_household),
        ],
        household_ids=[
            matching_household,
            non_matching_household,
        ],
        household_devices={
            matching_household: {"ev": True},
            non_matching_household: {"ev": True},
        },
        household_load_profile_kw={
            matching_household: [1.0, 2.0],
            non_matching_household: [3.0, 4.0],
        },
        ev_availability={
            matching_household: [1, 1],
            non_matching_household: [0, 0],
        },
    )

    filtered = filter_network_by_transformer(
        net,
        "Transformer_lv_grid_5690200024_1",
    )

    assert filtered.household_bus_ids == [matching_household]

    assert {str(bus.bus_id) for bus in filtered.buses} == {
        "hv_bus",
        "lv_bus",
        matching_household,
    }

    assert filtered.household_devices == {
        matching_household: {"ev": True},
    }

    assert filtered.household_load_profile_kw == {
        matching_household: [1.0, 2.0],
    }

    assert filtered.ev_availability == {
        matching_household: [1, 1],
    }

    assert len(filtered.transformers) == 1
    assert filtered.transformers[0].trafo_id == "Transformer_lv_grid_5690200024_1"