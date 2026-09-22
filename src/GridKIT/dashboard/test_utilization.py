# dashboard/test_utilization.py
import numpy as np

from dashboard import theme
from dashboard.utilization import (
    _daily_profile_chart,
    contiguous_blocks,
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


# ── Time axis ────────────────────────────────────────────────
def test_hours_axis_starts_at_noon_in_quarter_steps():
    axis = hours_axis(5)
    assert list(axis) == [12.0, 12.25, 12.5, 12.75, 13.0]


def test_format_hour_wraps_past_midnight():
    # the episode runs noon -> noon, so hour 25.5 is 01:30 the next day
    assert format_hour(25.5) == "01:30"
    assert format_hour(24.0) == "00:00"
    assert format_hour(14.25) == "14:15"


# ── Thresholds agree with the backend ────────────────
def test_exactly_rated_is_not_yet_overload():
    # run_experiment flags overload as `loading > 1.0`, strictly. If 1.0 were
    # already red the chart would contradict the overload matrix.
    assert theme.status_of(1.0) == "grenzbereich"
    assert theme.status_of(1.0001) == "kritisch"


def test_status_bands_cover_the_range():
    assert theme.status_of(0.0) == "gut"
    assert theme.status_of(0.7) == "gut"          # upper bound inclusive
    assert theme.status_of(0.71) == "warnung"
    assert theme.status_of(0.95) == "grenzbereich"
    assert theme.status_of(3.0) == "kritisch"


# ── Utilisation series ───────────────────────────────────────
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


# ── Overload matrix (Überlast-Matrix) ───────────────────
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
    assert row["Dauer gesamt (h)"] == 0.75
    assert row["Abschnitte"] == 1
    assert row["Zeitfenster"] == "12:30–13:00"


def test_total_duration_is_the_sum_not_the_span():
    # the reading that confused a reader: an element over the limit at the start
    # and again much later spans hours while being overloaded for minutes
    steps = [[] for _ in range(20)]
    steps[1] = ["service_2"]        # 12:15
    steps[18] = ["service_2"]       # 16:30
    tl = _timeline([0.3] * 20, [1.4] * 20, overloaded_lines=steps)
    row = overload_table(tl).iloc[0]
    assert row["Dauer gesamt (h)"] == 0.5          # two quarter hours, not four hours
    assert row["Zeitfenster"] == "12:15–16:30"     # the span is much wider
    assert row["Abschnitte"] == 2


def test_table_counts_separate_stretches():
    steps = [[] for _ in range(12)]
    for i in (1, 2, 3, 7, 10):
        steps[i] = ["service_0"]
    tl = _timeline([0.3] * 12, [1.4] * 12, overloaded_lines=steps)
    row = overload_table(tl).iloc[0]
    assert row["Abschnitte"] == 3
    assert row["Dauer gesamt (h)"] == 1.25


def test_table_reports_the_longest_single_stretch():
    # the figure that says whether a cable ever got time to cool down
    steps = [[] for _ in range(14)]
    for i in (0, 4, 5, 6, 7, 12):
        steps[i] = ["service_0"]
    tl = _timeline([0.3] * 14, [1.4] * 14, overloaded_lines=steps)
    row = overload_table(tl).iloc[0]
    assert row["Dauer davon (h)"] == 1.0            # steps 4..7
    assert row["längster Abschnitt"] == "13:00–13:45"


def test_single_step_stretch_is_written_as_one_time():
    steps = [[] for _ in range(4)]
    steps[2] = ["service_0"]
    tl = _timeline([0.3] * 4, [1.4] * 4, overloaded_lines=steps)
    row = overload_table(tl).iloc[0]
    assert row["längster Abschnitt"] == "12:30"     # not "12:30–12:30"
    assert row["Dauer davon (h)"] == 0.25


# ── Contiguous stretches ─────────────────────────────────────
def test_contiguous_blocks_splits_on_gaps():
    assert contiguous_blocks([0, 1, 2, 5, 6, 9]) == [(0, 2), (5, 6), (9, 9)]


def test_contiguous_blocks_of_one_run():
    assert contiguous_blocks([3, 4, 5]) == [(3, 5)]


def test_contiguous_blocks_of_nothing():
    assert contiguous_blocks([]) == []


def test_overload_table_includes_transformers_not_just_lines():
    tl = _timeline([1.2] * 2, [0.4] * 2, overloaded_lines=[[], []])
    tl["overloaded_transformers"] = [["trafo_1"], ["trafo_1"]]
    table = overload_table(tl)
    assert table.iloc[0]["Typ"] == "Trafo"


# ── Headline numbers ─────────────────────────────────────────
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


# ── Selection ────────────────────────────────────────────────
def test_select_timeline_matches_scenario_and_penetration():
    a = _timeline([0.1], [0.1]); a["penetration"] = 0.2
    b = _timeline([0.9], [0.9]); b["penetration"] = 0.6
    got = select_timeline([a, b], "1: flat / immediate", 0.6)
    assert got is b


def test_select_timeline_returns_none_when_absent():
    assert select_timeline([_timeline([0.1], [0.1])], "3: selfish RL", 0.6) is None


# ── Legend of the daily profile ──────────────────────────────
def _profile_spec():
    return _daily_profile_chart(_timeline([0.5, 0.8, 0.6], [1.2, 2.0, 1.4])).to_dict()


def test_daily_profile_labels_its_two_series():
    # the chart carries two series that mean entirely different things; without
    # a legend the reader cannot tell the cable from the transformer
    spec = _profile_spec()
    legends = [layer["encoding"]["color"].get("legend")
               for layer in spec["layer"]
               if layer.get("encoding", {}).get("color", {}).get("field") == "Messgröße"]
    assert legends and legends[0] is not None


def test_daily_profile_resolves_colour_independently():
    # a layered chart shares one colour scale by default, and the band layer
    # sets scale=None/legend=None, which silently removed the line legend
    assert spec_resolve(_profile_spec()) == "independent"


def spec_resolve(spec):
    return (spec.get("resolve") or {}).get("scale", {}).get("color")


def test_daily_profile_legend_names_both_series_with_theme_colours():
    spec = _profile_spec()
    colour = next(layer["encoding"]["color"] for layer in spec["layer"]
                  if layer.get("encoding", {}).get("color", {}).get("field") == "Messgröße")
    assert colour["scale"]["domain"] == [theme.SERIES_LABELS_DE["transformer"],
                                         theme.SERIES_LABELS_DE["line"]]
    assert colour["scale"]["range"] == [theme.SERIES_COLORS["transformer"],
                                        theme.SERIES_COLORS["line"]]
