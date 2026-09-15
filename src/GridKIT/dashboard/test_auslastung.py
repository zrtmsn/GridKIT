# dashboard/test_auslastung.py
import numpy as np

from dashboard import theme
from dashboard.auslastung import (
    duration_curve,
    format_hour,
    headline,
    hours_above,
    hours_axis,
    overload_matrix,
    overload_table,
    peak_moment,
    select_timeline,
    utilization_frame,
)


def _timeline(trafo, line, overloaded_lines=None, curtailment=None, **extra):
    n = len(trafo)
    return {
        "scenario": "1: flat / immediate",
        "penetration": 0.6,
        "transformer_loading": trafo,
        "max_line_loading": line,
        "curtailment": curtailment if curtailment is not None else [False] * n,
        "overloaded_lines": overloaded_lines if overloaded_lines is not None else [[] for _ in range(n)],
        "overloaded_transformers": [[] for _ in range(n)],
        **extra,
    }


# ── Zeitachse ────────────────────────────────────────────────
def test_hours_axis_starts_at_noon_in_quarter_steps():
    axis = hours_axis(5)
    assert list(axis) == [12.0, 12.25, 12.5, 12.75, 13.0]


def test_format_hour_wraps_past_midnight():
    # the episode runs noon -> noon, so hour 25.5 is 01:30 the next day
    assert format_hour(25.5) == "01:30"
    assert format_hour(24.0) == "00:00"
    assert format_hour(14.25) == "14:15"


# ── Schwellen stimmen mit dem Backend überein ────────────────
def test_exactly_rated_is_not_yet_overload():
    # run_experiment flags overload as `loading > 1.0`, strictly. If 1.0 were
    # already red the chart would contradict the overload matrix.
    assert theme.status_of(1.0) == "grenzbereich"
    assert theme.status_of(1.0001) == "kritisch"


def test_status_bands_cover_the_range():
    assert theme.status_of(0.0) == "gut"
    assert theme.status_of(0.7) == "gut"          # obere Grenze einschließend
    assert theme.status_of(0.71) == "warnung"
    assert theme.status_of(0.95) == "grenzbereich"
    assert theme.status_of(3.0) == "kritisch"


# ── Auslastungsreihen ────────────────────────────────────────
def test_utilization_frame_is_percent_not_pu():
    frame = utilization_frame(_timeline([0.5, 0.8], [1.2, 0.4]))
    trafo = frame[frame["reihe"] == "transformer"]["Auslastung"].tolist()
    assert trafo == [50.0, 80.0]


def test_utilization_frame_has_both_series_per_step():
    frame = utilization_frame(_timeline([0.5] * 4, [0.9] * 4))
    assert len(frame) == 8
    assert set(frame["reihe"]) == {"transformer", "line"}


def test_hours_above_counts_quarter_hours_strictly_above():
    # 3 steps over 1.0 -> 0.75 h; the exactly-1.0 step must NOT count
    assert hours_above([0.5, 1.01, 1.5, 1.0, 2.0]) == 0.75


def test_peak_moment_reports_value_and_clock_time():
    # peak at index 24 -> 12:00 + 6 h = 18:00
    values = [0.2] * 30
    values[24] = 1.12
    peak, at = peak_moment(values)
    assert round(peak, 1) == 112.0
    assert at == "18:00"


def test_peak_moment_handles_empty():
    # the placeholder is spelled out rather than a dash: a dash in a numeric
    # column also reads as a minus or as a measured zero
    assert peak_moment([]) == (0.0, theme.NO_VALUE)


# ── Dauerlinie ───────────────────────────────────────────────
def test_duration_curve_is_sorted_descending():
    curve = duration_curve([0.2, 1.0, 0.5])
    assert curve["Auslastung"].tolist() == [100.0, 50.0, 20.0]


def test_duration_curve_x_axis_is_cumulative_hours():
    curve = duration_curve([0.9] * 4)
    assert curve["Stunden"].tolist() == [0.25, 0.5, 0.75, 1.0]


def test_duration_curve_reads_off_hours_above_a_limit():
    # 8 quarter-hours over 100 % must show up as 2.0 h on the curve
    values = [1.5] * 8 + [0.4] * 88
    curve = duration_curve(values)
    over = curve[curve["Auslastung"] > 100.0]
    assert over["Stunden"].max() == 2.0


# ── Überlast-Matrix ──────────────────────────────────────────
def test_overload_matrix_marks_only_the_steps_an_element_tripped():
    tl = _timeline([0.3] * 4, [1.4] * 4,
                   overloaded_lines=[[], ["service_0"], ["service_0"], []])
    matrix = overload_matrix(tl)
    flags = matrix[matrix["Element"] == "service_0"]["überlastet"].tolist()
    assert flags == [False, True, True, False]


def test_overload_matrix_is_empty_when_nothing_tripped():
    assert overload_matrix(_timeline([0.3] * 4, [0.4] * 4)).empty


def test_overload_matrix_orders_worst_element_first():
    tl = _timeline([0.3] * 4, [1.4] * 4,
                   overloaded_lines=[["a"], ["a", "b"], ["a", "b"], ["a"]])
    order = list(dict.fromkeys(overload_matrix(tl)["Element"]))
    assert order[0] == "a"      # 4 steps beats b's 2


def test_overload_table_reports_duration_and_window():
    tl = _timeline([0.3] * 8, [1.4] * 8,
                   overloaded_lines=[[], [], ["service_0"], ["service_0"], ["service_0"], [], [], []])
    row = overload_table(tl).iloc[0]
    assert row["Element"] == "service_0"
    assert row["Dauer (h)"] == 0.75
    assert row["von"] == "12:30"     # step 2
    assert row["bis"] == "13:00"     # step 4


def test_overload_table_includes_transformers_not_just_lines():
    tl = _timeline([1.2] * 2, [0.4] * 2, overloaded_lines=[[], []])
    tl["overloaded_transformers"] = [["trafo_1"], ["trafo_1"]]
    table = overload_table(tl)
    assert table.iloc[0]["Typ"] == "Trafo"


# ── Kopfzahlen ───────────────────────────────────────────────
def test_headline_flags_the_cable_when_the_transformer_looks_calm():
    # the reference-run situation: transformer relaxed, cable far over limit
    tl = _timeline([0.8] * 10, [2.95] * 10,
                   overloaded_lines=[["service_0"]] * 10)
    head = headline(tl)
    assert head["status"] == "kritisch"
    assert round(head["trafo_peak_percent"]) == 80
    assert round(head["line_peak_percent"]) == 295
    assert head["worst_element"] == "service_0"


def test_headline_counts_curtailment_steps():
    tl = _timeline([0.3] * 4, [0.4] * 4, curtailment=[True, False, True, True])
    assert headline(tl)["curtailment_steps"] == 3


def test_headline_on_a_healthy_grid_is_not_critical():
    head = headline(_timeline([0.3] * 96, [0.45] * 96))
    assert head["status"] == "gut"
    assert head["hours_over"] == 0.0
    assert head["worst_element"] is None


# ── Auswahl ──────────────────────────────────────────────────
def test_select_timeline_matches_scenario_and_penetration():
    a = _timeline([0.1], [0.1]); a["penetration"] = 0.2
    b = _timeline([0.9], [0.9]); b["penetration"] = 0.6
    got = select_timeline([a, b], "1: flat / immediate", 0.6)
    assert got is b


def test_select_timeline_returns_none_when_absent():
    assert select_timeline([_timeline([0.1], [0.1])], "3: selfish RL", 0.6) is None
