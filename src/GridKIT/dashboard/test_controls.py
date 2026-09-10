# dashboard/test_controls.py
from dashboard.controls import available_penetrations, available_scenarios


def _tl(scenario, penetration):
    return {"scenario": scenario, "penetration": penetration}


def test_scenarios_come_back_in_the_projects_canonical_order():
    # data order is arbitrary; the picker must not be
    timelines = [
        _tl("3: selfish RL", 0.6),
        _tl("1: flat / immediate", 0.6),
        _tl("2: price-follow (manual)", 0.6),
    ]
    assert available_scenarios(timelines) == [
        "1: flat / immediate",
        "2: price-follow (manual)",
        "3: selfish RL",
    ]


def test_unknown_scenarios_are_appended_not_dropped():
    # a renamed or experimental scenario must stay reachable
    timelines = [_tl("4: experimental", 0.6), _tl("1: flat / immediate", 0.6)]
    got = available_scenarios(timelines)
    assert got[0] == "1: flat / immediate"
    assert "4: experimental" in got


def test_scenarios_are_deduplicated_across_penetrations():
    timelines = [_tl("1: flat / immediate", p) for p in (0.2, 0.4, 0.6)]
    assert available_scenarios(timelines) == ["1: flat / immediate"]


def test_scenarios_ignore_records_without_one():
    assert available_scenarios([{"penetration": 0.2}]) == []


def test_penetrations_are_sorted_and_unique():
    timelines = [_tl("a", 0.6), _tl("b", 0.2), _tl("c", 0.6), _tl("d", 0.4)]
    assert available_penetrations(timelines) == [0.2, 0.4, 0.6]


def test_empty_input_gives_empty_options():
    assert available_scenarios([]) == []
    assert available_penetrations([]) == []
