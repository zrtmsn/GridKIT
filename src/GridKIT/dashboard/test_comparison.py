# dashboard/test_comparison.py
from dashboard.comparison import METRICS, curtailment_steps, episode_frame, metric_frame


def _summary(scenario, pen, **over):
    record = {
        "scenario": scenario, "penetration": pen,
        "curtailment_mean": 30.0, "curtailment_std": 5.0,
        "soc_mean": 0.9, "soc_std": 0.05,
        "hp_comfort_mean": 1.0, "hp_comfort_std": 0.0,
        "bill_mean": 23.0, "bill_std": 2.0,
    }
    record.update(over)
    return record


def _timeline(n=4, **over):
    tl = {
        "scenario": "1: flat / immediate", "penetration": 0.6,
        "transformer_loading": [0.8] * n,
        "max_line_loading": [1.4] * n,
        "price": [0.3] * n,
        "pv_generation": [2.0] * n,
        "ev_power": [5.0] * n,
        "battery_power": [-1.0] * n,
        "hp_power": [3.0] * n,
        "curtailment": [False] * n,
    }
    tl.update(over)
    return tl


# ── Metrics (Kennzahlen je Szenario) ─────────────────────────
def test_metric_frame_has_a_row_per_scenario_and_metric():
    summary = [_summary("1: flat / immediate", 0.6), _summary("3: selfish RL", 0.6)]
    frame = metric_frame(summary, 0.6)
    assert len(frame) == 2 * len(METRICS)


def test_metric_frame_filters_to_one_ausstattungsgrad():
    summary = [_summary("a", 0.2), _summary("a", 0.6)]
    assert len(metric_frame(summary, 0.6)) == len(METRICS)


def test_metric_frame_keeps_the_spread_next_to_the_mean():
    frame = metric_frame([_summary("a", 0.6)], 0.6)
    row = frame[frame["kennzahl"] == "curtailment_mean"].iloc[0]
    assert row["Wert"] == 30.0
    assert row["Streuung"] == 5.0


def test_metric_frame_records_which_direction_is_better():
    # a reader cannot tell from a bar alone whether long is good
    frame = metric_frame([_summary("a", 0.6)], 0.6)
    by = dict(zip(frame["kennzahl"], frame["besser"]))
    assert by["curtailment_mean"] == "niedrig"
    assert by["soc_mean"] == "hoch"


def test_metric_frame_skips_fields_an_older_run_lacks():
    # summaries written before the producers were unified have no HP comfort
    record = _summary("a", 0.6)
    del record["hp_comfort_mean"]
    frame = metric_frame([record], 0.6)
    assert "WP-Komfort" not in set(frame["Kennzahl"])
    assert len(frame) == len(METRICS) - 1


def test_metric_frame_uses_german_scenario_labels():
    frame = metric_frame([_summary("3: selfish RL", 0.6)], 0.6)
    assert frame["Szenario"].iloc[0] == "3: eigennütziges RL"


def test_metric_frame_of_nothing_is_empty_but_typed():
    frame = metric_frame([], 0.6)
    assert frame.empty and "Wert" in frame.columns


# ── Episode ──────────────────────────────────────────────────
def test_episode_frame_groups_series_by_axis():
    frame = episode_frame(_timeline(n=4))
    assert set(frame["Gruppe"]) == {"Auslastung", "Leistung", "Preis"}


def test_episode_frame_converts_loading_to_percent():
    frame = episode_frame(_timeline(n=2))
    trafo = frame[frame["reihe"] == "transformer_loading"]["Wert"].tolist()
    assert trafo == [80.0, 80.0]


def test_episode_frame_leaves_price_and_power_unscaled():
    frame = episode_frame(_timeline(n=2))
    assert frame[frame["reihe"] == "price"]["Wert"].tolist() == [0.3, 0.3]
    assert frame[frame["reihe"] == "ev_power"]["Wert"].tolist() == [5.0, 5.0]


def test_episode_frame_keeps_the_battery_sign():
    frame = episode_frame(_timeline(n=2, battery_power=[2.0, -2.0]))
    assert frame[frame["reihe"] == "battery_power"]["Wert"].tolist() == [2.0, -2.0]


def test_episode_frame_starts_at_noon():
    frame = episode_frame(_timeline(n=2))
    assert frame["Uhrzeit"].iloc[0] == "12:00"


def test_episode_frame_of_nothing_is_empty_but_typed():
    frame = episode_frame({})
    assert frame.empty and "Gruppe" in frame.columns


# ── Curtailment (Abregelung) ─────────────────────────────────
def test_curtailment_steps_lists_only_the_active_quarter_hours():
    frame = curtailment_steps(_timeline(n=4, curtailment=[False, True, True, False]))
    assert frame["Uhrzeit"].tolist() == ["12:15", "12:30"]


def test_curtailment_steps_is_empty_when_nothing_was_curtailed():
    assert curtailment_steps(_timeline(n=4)).empty
