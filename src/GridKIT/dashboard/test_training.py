# dashboard/test_training.py
import json

from dashboard import theme
from dashboard.training import (
    METRICS_FILENAME,
    SINGLE_RUN,
    convergence,
    convergence_domain,
    entropy_frame,
    evaluated_rewards,
    final_entropy,
    find_metric_files,
    iterations_outside,
    load_metrics,
    penetration_of,
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
def _sweep(root, *pens):
    """Batch-experiment layout: checkpoints/pen_XX/iteration_metrics.json."""
    for pen in pens:
        d = root / "checkpoints" / pen
        d.mkdir(parents=True)
        (d / METRICS_FILENAME).write_text("[]", encoding="utf-8")


def test_find_metric_files_picks_up_each_penetration(tmp_path):
    _sweep(tmp_path, "pen_20", "pen_40")
    assert sorted(find_metric_files(tmp_path)) == ["pen_20", "pen_40"]


def test_find_metric_files_skips_penetrations_without_metrics(tmp_path):
    # an interrupted sweep leaves a checkpoint dir with no metrics file
    (tmp_path / "checkpoints" / "pen_20").mkdir(parents=True)
    _sweep(tmp_path, "pen_40")
    assert list(find_metric_files(tmp_path)) == ["pen_40"]


def test_find_metric_files_on_missing_dir_is_empty(tmp_path):
    assert find_metric_files(tmp_path / "nope") == {}


def test_finds_metrics_of_a_run_started_from_the_map(tmp_path):
    # train_run.py trains one device layout and writes the metrics at the run
    # root, with no pen_* level; the training tab used to come up empty for
    # exactly the runs a user creates by drawing an area
    (tmp_path / METRICS_FILENAME).write_text("[]", encoding="utf-8")
    (tmp_path / "checkpoints").mkdir()
    assert list(find_metric_files(tmp_path)) == [SINGLE_RUN]


def test_a_sweep_wins_over_a_stray_root_file(tmp_path):
    _sweep(tmp_path, "pen_20")
    (tmp_path / METRICS_FILENAME).write_text("[]", encoding="utf-8")
    assert list(find_metric_files(tmp_path)) == ["pen_20"]


def test_single_run_key_has_no_penetration():
    # nothing in its path says which share it was, so the tab must not pretend
    assert penetration_of(SINGLE_RUN) is None


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


# ── Zoom auf die konvergierte Phase ──────────────────────────
def _run(values):
    """One record per value, with a spread of ±10 around it."""
    return [_record(i + 1, v, v - 10.0, v + 10.0) for i, v in enumerate(values)]


def test_convergence_domain_ignores_the_catastrophic_start():
    # the shape of the reference run: one enormous early value, then a plateau
    frame = return_frame(_run([-13000.0] + [-900.0] * 15))
    low, high = convergence_domain(frame)
    assert low > -2000.0, "the untrained iteration must not set the axis"
    assert high > -900.0


def test_convergence_domain_leaves_the_converged_spread_visible():
    frame = return_frame(_run([-13000.0] + [-1100.0, -900.0] * 8))
    low, high = convergence_domain(frame)
    assert low < -1110.0 and high > -890.0


def test_convergence_domain_of_a_short_run_is_none():
    # with a handful of iterations the whole curve is warm-up; cropping it
    # would leave an empty chart
    assert convergence_domain(return_frame(_run([-900.0] * 5))) is None


def test_convergence_domain_of_a_flat_run_is_none():
    # identical returns with no spread give a zero-height domain, which would
    # collapse the chart; fall back to the automatic axis instead
    frame = return_frame([_record(i, -900.0) for i in range(1, 13)])
    assert convergence_domain(frame) is None


def test_convergence_domain_of_nothing_is_none():
    assert convergence_domain(return_frame([])) is None


def test_iterations_outside_counts_the_cropped_points():
    frame = return_frame(_run([-13000.0, -12000.0] + [-900.0] * 14))
    assert iterations_outside(frame, convergence_domain(frame)) == 2


def test_iterations_outside_without_a_domain_is_zero():
    assert iterations_outside(return_frame(_run([-900.0] * 12)), None) == 0


# ── Reward je Strategie ──────────────────────────────────────
def _summary_row(scenario, reward, std=1.0, penetration=0.6):
    return {"scenario": scenario, "penetration": penetration,
            "reward_mean": reward, "reward_std": std}


def test_evaluated_rewards_sorts_best_first():
    frame = evaluated_rewards([
        _summary_row("2: price-follow (manual)", -34.1),
        _summary_row("1: flat / immediate", -23.3),
        _summary_row("3: selfish RL", -25.9),
    ])
    assert frame["scenario"].tolist()[0] == "1: flat / immediate"
    assert frame["Reward"].is_monotonic_decreasing


def test_evaluated_rewards_marks_the_learned_policy():
    frame = evaluated_rewards([_summary_row("1: flat / immediate", -23.3),
                               _summary_row("3: selfish RL", -25.9)])
    learned = frame[frame["gelernt"]]
    assert len(learned) == 1
    assert learned.iloc[0]["scenario"] == "3: selfish RL"


def test_evaluated_rewards_uses_german_scenario_labels():
    frame = evaluated_rewards([_summary_row("3: selfish RL", -25.9)])
    assert frame["Szenario"].iloc[0] == theme.SCENARIO_LABELS_DE["3: selfish RL"]


def test_evaluated_rewards_keeps_only_the_chosen_penetration():
    frame = evaluated_rewards([_summary_row("3: selfish RL", -25.9, penetration=0.6),
                               _summary_row("3: selfish RL", -40.0, penetration=0.2)],
                              penetration=0.6)
    assert frame["Reward"].tolist() == [-25.9]


def test_evaluated_rewards_without_a_penetration_takes_everything():
    frame = evaluated_rewards([_summary_row("3: selfish RL", -25.9, penetration=0.6),
                               _summary_row("1: flat / immediate", -23.3, penetration=0.2)])
    assert len(frame) == 2


def test_evaluated_rewards_skips_records_without_a_reward():
    frame = evaluated_rewards([{"scenario": "1: flat / immediate", "penetration": 0.6},
                               _summary_row("3: selfish RL", -25.9)])
    assert frame["scenario"].tolist() == ["3: selfish RL"]


def test_evaluated_rewards_defaults_a_missing_spread_to_zero():
    frame = evaluated_rewards([{"scenario": "3: selfish RL", "penetration": 0.6,
                                "reward_mean": -25.9, "reward_std": None}])
    assert frame["Streuung"].iloc[0] == 0.0


def test_evaluated_rewards_of_nothing_is_empty_but_typed():
    frame = evaluated_rewards([])
    assert frame.empty
    assert list(frame.columns) == ["scenario", "Szenario", "Reward", "Streuung", "gelernt"]
