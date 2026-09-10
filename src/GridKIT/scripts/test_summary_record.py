# scripts/test_summary_record.py
"""The summary.json contract.

run_experiment.py (batch sweep) and train_run.py (the web app's background
training) both write summary.json. They used to build the record separately and
drifted into different schemas — neither a superset of the other — so a
dashboard tile was empty for half the runs depending on how they were started.
These tests pin the union so the two cannot diverge again.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.run_experiment import summary_record

_SCRIPTS = Path(__file__).resolve().parent


class _Stats:
    """Minimal stand-in for ScenarioStats — only what summary_record reads."""
    curtailment_events = (12.0, 1.5)
    soc_satisfaction_rate = (0.94, 0.04)
    transformer_peak_loading_pu = (0.80, 0.05)
    mean_episode_reward = (-24.5, 1.6)
    mean_household_bill_eur = (24.5, 1.6)
    hp_comfort_satisfaction_rate = (0.97, 0.02)
    battery_charge_kwh = (60.0, 3.0)
    battery_discharge_kwh = (40.0, 4.0)
    feeder_overload_steps = {"feeder_1": (2.0, 0.5)}
    feeder_peak_loading_pu = {"feeder_1": (1.04, 0.03)}
    line_overload_steps = {"service_5": (8.0, 1.0), "service_0": (3.0, 0.5)}
    line_peak_loading_pu = {"service_5": (2.95, 0.2), "service_0": (1.4, 0.1)}

    def worst_feeder(self):
        return ("feeder_1", 2.0, 1.04)


@pytest.fixture
def record():
    return summary_record(_Stats(), "3: selfish RL", 0.6)


# ── the fields the dashboard is scoped to ────────────────────
DASHBOARD_FIELDS = [
    "penetration", "scenario",
    "curtailment_mean", "soc_mean", "soc_std", "peak_mean", "peak_std",
    "reward_mean", "reward_std", "bill_mean", "bill_std",
    "hp_comfort_mean", "hp_comfort_std",
    "battery_throughput_kwh_mean", "battery_throughput_kwh_std",
    "battery_full_cycles_mean", "battery_full_cycles_std",
]


def test_record_carries_every_field_the_dashboard_reads(record):
    missing = [f for f in DASHBOARD_FIELDS if f not in record]
    assert missing == []


def test_record_keeps_the_feeder_and_line_breakdown(record):
    # train_run.py's contribution to the union — the web app path had these,
    # the batch path did not
    for field in ("line_peak_max", "n_lines_overloaded", "worst_feeder",
                  "worst_feeder_steps", "feeder_overload_steps", "line_peak_loading_pu"):
        assert field in record


def test_scenario_and_penetration_are_passed_through(record):
    assert record["scenario"] == "3: selfish RL"
    assert record["penetration"] == 0.6


# ── the derived battery figures ──────────────────────────────
def test_throughput_adds_both_legs(record):
    assert record["battery_throughput_kwh_mean"] == pytest.approx(100.0)


def test_throughput_spread_combines_in_quadrature(record):
    # charge and discharge are aggregated separately, so their spreads do not
    # simply add: sqrt(3^2 + 4^2) == 5
    assert record["battery_throughput_kwh_std"] == pytest.approx(5.0)


def test_full_cycles_is_throughput_over_two_capacities(record):
    import core.constants as const
    assert record["battery_full_cycles_mean"] == pytest.approx(
        100.0 / 2.0 / const.BATTERY_CAPACITY_KWH
    )


def test_worst_feeder_is_unpacked_not_left_as_a_tuple(record):
    assert record["worst_feeder"] == "feeder_1"
    assert record["worst_feeder_steps"] == 2.0


def test_line_peak_max_is_the_worst_cable(record):
    assert record["line_peak_max"] == pytest.approx(2.95)


def test_dicts_are_flattened_to_the_mean_only(record):
    # (mean, std) tuples would not survive a JSON round trip as numbers
    assert record["line_overload_steps"] == {"service_5": 8.0, "service_0": 3.0}


def test_absent_worst_feeder_does_not_crash():
    class _NoFeeder(_Stats):
        def worst_feeder(self):
            return None

    record = summary_record(_NoFeeder(), "1: flat / immediate", 0.2)
    assert record["worst_feeder"] is None
    assert record["worst_feeder_steps"] == 0.0


# ── neither producer may hand-roll the record again ──────────
@pytest.mark.parametrize("script", ["run_experiment.py", "train_run.py"])
def test_producers_build_the_record_through_the_shared_helper(script):
    """Guards the actual regression: a hand-built dict drifting out of sync.

    Both files must reach summary.json through summary_record(), so a field
    added for one producer cannot silently go missing for the other.
    """
    tree = ast.parse((_SCRIPTS / script).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "summary"
    ]
    assert calls, f"{script} does not append to a `summary` list any more"
    for call in calls:
        assert call.args and isinstance(call.args[0], ast.Call), (
            f"{script} builds a summary row inline instead of calling summary_record()"
        )
        assert call.args[0].func.id == "summary_record", (
            f"{script} appends something other than summary_record(...)"
        )
