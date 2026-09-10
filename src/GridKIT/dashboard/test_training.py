# dashboard/test_training.py
import json

from dashboard import theme
from dashboard.training import (
    METRICS_FILENAME,
    convergence,
    entropy_frame,
    final_entropy,
    find_metric_files,
    load_metrics,
    return_frame,
)


def _record(iteration, mean, lo=None, hi=None, entropies=None):
    learners = {"__all_modules__": {"total_loss": 1.0}}
    for policy, value in (entropies or {}).items():
        learners[policy] = {"entropy": value, "policy_loss": -0.1}
    return {
        "training_iteration": iteration,
        "env_runners": {
            "episode_return_mean": mean,
            "episode_return_min": mean if lo is None else lo,
            "episode_return_max": mean if hi is None else hi,
        },
        "learners": learners,
    }


# ── Dateien finden ───────────────────────────────────────────
def test_find_metric_files_picks_up_each_penetration(tmp_path):
    for pen in ("pen_20", "pen_40"):
        d = tmp_path / pen
        d.mkdir()
        (d / METRICS_FILENAME).write_text("[]", encoding="utf-8")
    assert sorted(find_metric_files(tmp_path)) == ["pen_20", "pen_40"]


def test_find_metric_files_skips_penetrations_without_metrics(tmp_path):
    # an interrupted sweep leaves a checkpoint dir with no metrics file
    (tmp_path / "pen_20").mkdir()
    done = tmp_path / "pen_40"
    done.mkdir()
    (done / METRICS_FILENAME).write_text("[]", encoding="utf-8")
    assert list(find_metric_files(tmp_path)) == ["pen_40"]


def test_find_metric_files_on_missing_dir_is_empty(tmp_path):
    assert find_metric_files(tmp_path / "nope") == {}


def test_load_metrics_of_corrupt_file_is_empty_not_an_exception(tmp_path):
    # a half-written file from an interrupted run must not take the dashboard down
    path = tmp_path / METRICS_FILENAME
    path.write_text("[{'broken': ", encoding="utf-8")
    assert load_metrics(path) == []


def test_load_metrics_reads_a_list(tmp_path):
    path = tmp_path / METRICS_FILENAME
    path.write_text(json.dumps([_record(1, -100.0)]), encoding="utf-8")
    assert len(load_metrics(path)) == 1


# ── Return-Verlauf ───────────────────────────────────────────
def test_return_frame_keeps_iteration_order_and_spread():
    frame = return_frame([_record(1, -500.0, -600.0, -400.0), _record(2, -300.0, -350.0, -250.0)])
    assert frame["Iteration"].tolist() == [1, 2]
    assert frame["Mittel"].tolist() == [-500.0, -300.0]
    assert frame["Minimum"].tolist() == [-600.0, -350.0]


def test_return_frame_falls_back_to_mean_when_no_spread_recorded():
    frame = return_frame([_record(1, -500.0)])
    assert frame["Minimum"].iloc[0] == -500.0
    assert frame["Maximum"].iloc[0] == -500.0


def test_return_frame_skips_records_without_a_return():
    broken = {"training_iteration": 2, "env_runners": {}}
    frame = return_frame([_record(1, -100.0), broken])
    assert frame["Iteration"].tolist() == [1]


def test_return_frame_of_nothing_is_empty_but_typed():
    frame = return_frame([])
    assert frame.empty
    assert list(frame.columns) == ["Iteration", "Mittel", "Minimum", "Maximum"]


# ── Entropie ─────────────────────────────────────────────────
def test_entropy_frame_has_one_row_per_policy_and_iteration():
    records = [_record(i, -100.0, entropies={"ev_policy": 1.0, "hp_policy": 0.6,
                                             "battery_policy": 1.1}) for i in (1, 2)]
    frame = entropy_frame(records)
    assert len(frame) == 6
    assert set(frame["policy"]) == {"ev_policy", "hp_policy", "battery_policy"}


def test_entropy_frame_excludes_the_all_modules_aggregate():
    # RLlib writes __all_modules__ alongside the real policies; it is not a
    # policy and would plot as a meaningless fourth line
    frame = entropy_frame([_record(1, -100.0, entropies={"ev_policy": 1.0})])
    assert "__all_modules__" not in set(frame["policy"])


def test_entropy_frame_uses_german_device_labels():
    frame = entropy_frame([_record(1, -100.0, entropies={"hp_policy": 0.6})])
    assert frame["Policy"].iloc[0] == "Wärmepumpe"


def test_entropy_frame_without_entropy_is_empty():
    assert entropy_frame([_record(1, -100.0)]).empty


def test_final_entropy_takes_the_last_iteration_per_policy():
    records = [
        _record(1, -100.0, entropies={"ev_policy": 1.2, "hp_policy": 0.9}),
        _record(2, -90.0, entropies={"ev_policy": 0.8, "hp_policy": 0.3}),
    ]
    assert final_entropy(records) == {"ev_policy": 0.8, "hp_policy": 0.3}


def test_final_entropy_identifies_the_policy_that_stopped_exploring():
    records = [_record(1, -100.0, entropies={"ev_policy": 1.1, "hp_policy": 0.2,
                                             "battery_policy": 1.0})]
    entropies = final_entropy(records)
    assert min(entropies, key=entropies.get) == "hp_policy"


# ── Konvergenz ───────────────────────────────────────────────
def test_convergence_reports_the_change_across_the_run():
    conv = convergence([_record(1, -500.0), _record(2, -400.0), _record(3, -300.0)])
    assert conv["iterations"] == 3
    assert conv["first"] == -500.0
    assert conv["last"] == -300.0
    assert conv["delta"] == 200.0
    assert conv["improved"] is True


def test_convergence_notices_a_run_that_got_worse():
    conv = convergence([_record(1, -300.0), _record(2, -900.0)])
    assert conv["improved"] is False
    assert conv["delta"] == -600.0


def test_convergence_of_nothing_is_safe():
    conv = convergence([])
    assert conv["iterations"] == 0 and conv["last"] is None


# ── Policy-Beschriftung ──────────────────────────────────────
def test_policy_label_and_colour_match_the_device():
    assert theme.policy_label("battery_policy") == "Batterie"
    assert theme.policy_color("battery_policy") == theme.DEVICE_COLORS["battery"]


def test_unknown_policy_falls_back_to_its_id():
    assert theme.policy_label("mystery_policy") == "mystery_policy"
