# dashboard/test_ueberblick.py
import pandas as pd

from dashboard.ueberblick import (
    breaking_point,
    cable_peak_pu,
    headline_numbers,
    overload_hours,
    overview_frame,
    safe_ceiling,
    scenario_ranking,
    worst_status,
)


def _row(scenario, pen, peak_mean, line_peak_max=None, **extra):
    record = {"scenario": scenario, "penetration": pen, "peak_mean": peak_mean,
              "curtailment_mean": 10.0, "soc_mean": 0.9}
    if line_peak_max is not None:
        record["line_peak_max"] = line_peak_max
    record.update(extra)
    return record


def _timeline(scenario, pen, line_peaks):
    return {"scenario": scenario, "penetration": pen, "max_line_loading": line_peaks}


# ── Kabelspitze: neue Läufe direkt, alte über die Zeitreihe ──
def test_cable_peak_prefers_the_seed_averaged_field():
    record = _row("1: flat / immediate", 0.6, 0.8, line_peak_max=2.15)
    assert cable_peak_pu(record) == 2.15


def test_cable_peak_falls_back_to_the_representative_episode():
    # summaries written before the two producers were unified have no line_peak_max
    record = _row("1: flat / immediate", 0.6, 0.8)
    timelines = [_timeline("1: flat / immediate", 0.6, [0.4, 1.9, 1.2])]
    assert cable_peak_pu(record, timelines) == 1.9


def test_cable_peak_is_none_when_neither_source_has_it():
    assert cable_peak_pu(_row("x", 0.6, 0.8), []) is None


def test_cable_peak_fallback_matches_on_both_scenario_and_penetration():
    record = _row("3: selfish RL", 0.6, 0.8)
    timelines = [
        _timeline("3: selfish RL", 0.2, [9.0]),          # right scenario, wrong share
        _timeline("1: flat / immediate", 0.6, [8.0]),    # right share, wrong scenario
        _timeline("3: selfish RL", 0.6, [2.95]),
    ]
    assert cable_peak_pu(record, timelines) == 2.95


# ── Übersichtsmatrix ─────────────────────────────────────────
def test_status_follows_the_worse_of_transformer_and_cable():
    # a relaxed transformer must not hide a cable over its limit — the exact
    # misreading this dashboard exists to prevent
    frame = overview_frame([_row("3: selfish RL", 0.6, 0.80, line_peak_max=2.95)])
    assert frame["status"].iloc[0] == "kritisch"
    assert frame["Spitze"].iloc[0] == 2.95


def test_status_uses_the_transformer_when_it_is_the_worse_one():
    frame = overview_frame([_row("1: flat / immediate", 0.6, 1.10, line_peak_max=0.5)])
    assert frame["Spitze"].iloc[0] == 1.10
    assert frame["status"].iloc[0] == "kritisch"


def test_overview_frame_translates_scenario_labels():
    frame = overview_frame([_row("3: selfish RL", 0.2, 0.4)])
    assert frame["Szenario"].iloc[0] == "3: eigennütziges RL"
    assert frame["EV-Anteil"].iloc[0] == "20%"


def test_overview_frame_of_nothing_is_empty_but_typed():
    frame = overview_frame([])
    assert frame.empty and "status" in frame.columns


# ── Urteil über alle Kombinationen ───────────────────────────
def test_worst_status_picks_the_most_severe_present():
    frame = overview_frame([
        _row("a", 0.2, 0.3), _row("b", 0.4, 0.95), _row("c", 0.6, 1.4),
    ])
    assert worst_status(frame) == "kritisch"


def test_worst_status_of_a_healthy_run():
    frame = overview_frame([_row("a", 0.2, 0.3), _row("b", 0.4, 0.5)])
    assert worst_status(frame) == "gut"


# ── Belastungsgrenze ─────────────────────────────────────────
def test_breaking_point_is_the_lowest_share_that_overloads():
    frame = overview_frame([
        _row("a", 0.2, 0.5), _row("a", 0.4, 1.05), _row("a", 0.6, 1.4),
    ])
    assert breaking_point(frame) == 0.4


def test_breaking_point_is_none_when_nothing_overloads():
    assert breaking_point(overview_frame([_row("a", 0.2, 0.5)])) is None


def test_safe_ceiling_is_the_highest_share_where_every_scenario_holds():
    frame = overview_frame([
        _row("a", 0.2, 0.5), _row("b", 0.2, 0.6),
        _row("a", 0.4, 0.7), _row("b", 0.4, 0.8),
        _row("a", 0.6, 1.4), _row("b", 0.6, 0.9),
    ])
    # 0.6 has one scenario over the limit, so the ceiling is 0.4
    assert safe_ceiling(frame) == 0.4


def test_safe_ceiling_is_none_when_even_the_lowest_share_breaks():
    frame = overview_frame([_row("a", 0.2, 1.3), _row("a", 0.4, 1.5)])
    assert safe_ceiling(frame) is None


def test_one_bad_scenario_disqualifies_the_whole_penetration():
    frame = overview_frame([_row("good", 0.6, 0.4), _row("bad", 0.6, 1.9)])
    assert safe_ceiling(frame) is None


# ── Rangfolge ────────────────────────────────────────────────
def test_scenario_ranking_is_gentlest_first():
    frame = overview_frame([
        _row("hard", 0.6, 1.5), _row("soft", 0.6, 0.4), _row("mid", 0.6, 0.9),
        _row("other", 0.2, 0.1),
    ])
    ranked = scenario_ranking(frame, 0.6)
    assert ranked["scenario"].tolist() == ["soft", "mid", "hard"]


def test_scenario_ranking_only_covers_the_requested_share():
    frame = overview_frame([_row("a", 0.2, 0.4), _row("b", 0.6, 0.5)])
    assert scenario_ranking(frame, 0.6)["scenario"].tolist() == ["b"]


# ── Kopfzahlen ───────────────────────────────────────────────
def test_headline_reports_where_the_worst_case_was():
    frame = overview_frame([
        _row("1: flat / immediate", 0.2, 0.4),
        _row("3: selfish RL", 0.6, 0.80, line_peak_max=2.95),
    ])
    head = headline_numbers(frame)
    assert head["worst_peak"] == 2.95
    assert head["worst_scenario"] == "3: eigennütziges RL"
    assert head["worst_penetration"] == 0.6
    assert head["status"] == "kritisch"


def test_headline_counts_affected_combinations():
    frame = overview_frame([
        _row("a", 0.2, 0.4), _row("b", 0.6, 1.2), _row("c", 0.6, 1.4),
    ])
    head = headline_numbers(frame)
    assert head["n_over"] == 2 and head["n_total"] == 3


def test_headline_of_an_empty_run_is_safe():
    head = headline_numbers(overview_frame([]))
    assert head["status"] is None and head["n_total"] == 0


def test_headline_survives_rows_without_any_loading():
    frame = overview_frame([{"scenario": "a", "penetration": 0.2}])
    assert headline_numbers(frame)["status"] is None


# ── Dauer statt Spitze: was die Szenarien wirklich trennt ────
def test_overload_hours_uses_the_longest_affected_element():
    # max, not sum: two cables over the limit in the same quarter hour is one
    # overloaded quarter hour, not two
    record = _row("a", 0.6, 1.2,
                  line_overload_steps={"service_0": 8.0, "service_5": 36.0},
                  feeder_overload_steps={"feeder_1": 3.0})
    assert overload_hours(record) == 9.0        # 36 steps x 0.25 h


def test_overload_hours_covers_feeders_as_well_as_lines():
    record = _row("a", 0.6, 1.2, feeder_overload_steps={"feeder_1": 12.0})
    assert overload_hours(record) == 3.0


def test_overload_hours_falls_back_to_the_episode_for_older_runs():
    record = _row("a", 0.6, 1.2)
    timelines = [{"scenario": "a", "penetration": 0.6,
                  "max_line_loading": [1.5, 1.5, 0.4, 1.5],
                  "transformer_loading": [0.5] * 4}]
    assert overload_hours(record, timelines) == 0.75     # 3 steps


def test_overload_hours_is_none_without_any_source():
    assert overload_hours(_row("a", 0.6, 1.2), []) is None


def test_overload_hours_of_a_clean_run_is_zero_not_none():
    record = _row("a", 0.2, 0.5, line_overload_steps={}, feeder_overload_steps={})
    timelines = [{"scenario": "a", "penetration": 0.2,
                  "max_line_loading": [0.4] * 4, "transformer_loading": [0.3] * 4}]
    assert overload_hours(record, timelines) == 0.0


def test_ranking_orders_by_duration_not_peak():
    # the case that motivated this: one weak cable pins every peak to ~2.0, so
    # ranking on peak would order the scenarios essentially at random, while
    # the time spent in overload separates them clearly
    frame = overview_frame([
        _row("rl", 0.6, 0.78, line_peak_max=1.98, line_overload_steps={"s": 36.0}),
        _row("flat", 0.6, 1.02, line_peak_max=2.00, line_overload_steps={"s": 42.0}),
    ])
    assert scenario_ranking(frame, 0.6)["scenario"].tolist() == ["rl", "flat"]


def test_ranking_still_works_when_no_duration_is_available():
    frame = overview_frame([_row("hard", 0.6, 1.5), _row("soft", 0.6, 0.4)])
    assert scenario_ranking(frame, 0.6)["scenario"].tolist() == ["soft", "hard"]


def test_headline_names_the_longest_overload_not_the_highest_reading():
    frame = overview_frame([
        # highest peak, but brief
        _row("spiky", 0.6, 0.80, line_peak_max=2.80, line_overload_steps={"s": 4.0}),
        # lower peak, but overloaded far longer — this is the one to name
        _row("long", 0.4, 1.05, line_peak_max=1.20, line_overload_steps={"s": 40.0}),
    ])
    head = headline_numbers(frame)
    assert head["worst_hours"] == 10.0
    assert head["worst_hours_scenario"] == "long"
    assert head["worst_hours_penetration"] == 0.4
    # the peak is still reported, just no longer the headline
    assert head["worst_peak"] == 2.80


def test_headline_hours_are_absent_when_no_run_records_them():
    frame = overview_frame([_row("a", 0.6, 1.2)])
    assert headline_numbers(frame)["worst_hours"] is None
