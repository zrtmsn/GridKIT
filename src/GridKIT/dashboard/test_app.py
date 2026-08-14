# dashboard/test_app.py
import pandas as pd

import core.constants as const
from GridKIT.dashboard.app import (
    _LOAD_COLORS,
    _load_style,
    _map_caption,
    _penetration_levels,
    _recorded_rows,
    LINE_PEAK_KEY,
    LINE_WATCH_PU,
)


def test_overloaded_cable_is_red_and_thickest():
    color, weight = _load_style(1.05)
    assert color == "#d7191c"
    assert weight == max(w for _, _, w in _LOAD_COLORS)


def test_severity_bands_are_distinct():
    bands = [_load_style(pu)[0] for pu in (1.20, 0.95, 0.80, 0.20)]
    assert len(set(bands)) == 4, "each severity band needs its own colour"


def test_weight_increases_with_loading():
    weights = [_load_style(pu)[1] for pu in (0.2, 0.8, 0.95, 1.2)]
    assert weights == sorted(weights)


def test_exactly_at_the_overload_threshold_is_not_yet_red():
    # the environment curtails on `> threshold`, so the map must agree — a cable
    # sitting exactly at rating has not tripped
    color, _ = _load_style(const.LINE_OVERLOAD_THRESHOLD)
    assert color != "#d7191c"


def test_watch_level_still_renders_a_colour():
    # anything recorded at all is drawn coloured; grey is reserved for lines the
    # runner never recorded (below the watch level)
    color, _ = _load_style(const.LINE_WATCH_THRESHOLD)
    assert color.startswith("#")


def test_watch_level_matches_the_runner():
    # the map's "below watch level" grey must mean exactly what the runner declined to record
    assert LINE_WATCH_PU == const.LINE_WATCH_THRESHOLD


# ── which runs can be mapped ──────────────────────────────────
def test_a_quiet_run_is_still_mappable():
    # key present, dict empty = every cable stayed under the watch level. That is a
    # result, not a missing feature — it must not be filtered out as an old run.
    rows = _recorded_rows([{"scenario": "3: selfish RL", LINE_PEAK_KEY: {}}])
    assert len(rows) == 1


def test_a_run_from_before_per_line_recording_is_not_mappable():
    rows = _recorded_rows([{"scenario": "1: flat / immediate", "curtailment_mean": 0.0}])
    assert rows == []


def test_recorded_rows_tolerates_missing_summary():
    assert _recorded_rows(None) == []


def test_quiet_caption_states_the_finding_not_an_absence():
    caption = _map_caption("3: selfish RL", {}, 0)
    assert f"{LINE_WATCH_PU:.1f} pu" in caption
    assert "0.00 pu" not in caption, "a quiet grid must not be reported as a 0.00 pu worst cable"


def test_loaded_caption_reports_the_worst_cable_and_trip_count():
    caption = _map_caption("2: price-follow (automated)", {"l1": 0.62, "l2": 1.08}, 1)
    assert "1.08 pu" in caption
    assert "1 cable(s)" in caption


# ── penetration is a sweep axis, not a label ──────────────────
def test_a_designed_run_has_no_penetration_axis():
    # every row of one designed run carries the same derived figure — not a comparison
    df = pd.DataFrame([{"scenario": "1: flat / immediate", "penetration": 0.63},
                       {"scenario": "3: selfish RL", "penetration": 0.63}])
    assert len(_penetration_levels(df)) == 1


def test_the_batch_experiment_still_sweeps_penetration():
    df = pd.DataFrame([{"scenario": "3: selfish RL", "penetration": p}
                       for p in const.EV_PENETRATION_LEVELS])
    assert _penetration_levels(df) == sorted(const.EV_PENETRATION_LEVELS)


def test_summary_without_the_field_is_not_a_sweep():
    assert _penetration_levels(pd.DataFrame([{"scenario": "3: selfish RL"}])) == []


def test_a_cable_that_tripped_reads_as_overloaded_even_when_its_mean_is_lower():
    # peaks are averaged over seeds, so a cable that violated its rating in a minority
    # of seeds sits below 1.0 on average — the map must not contradict the trip count
    plain, _ = _load_style(0.85)
    tripped, weight = _load_style(0.85, tripped=True)
    assert plain != tripped
    assert tripped == "#d7191c"
    assert weight == max(w for _, _, w in _LOAD_COLORS)
