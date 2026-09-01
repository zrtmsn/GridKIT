# scripts/train_run.py
# ─────────────────────────────────────────────────────────────
# Background trainer for ONE run saved via core.run_store (a network +
# household_configuration.json from the map UI).
#
# Loads the run's grid_network.json + household_configuration.json, trains an
# IPPO policy on that exact grid + device layout, evaluates it against the
# rule-based baselines, and writes the result via run_store.save_results() —
# updating run_store status live so the UI can show progress while it trains.
#
# Training hyperparameters (iterations/seeds) are FIXED
# (core.constants.PIPELINE_TRAINING_ITERATIONS/PIPELINE_EVALUATION_SEEDS), not
# read from the run config — the UI never exposes RL knobs to the user, so
# there is nothing per-run to read here.
#
# Launched detached by map_ui/map_widget.py:
#   PYTHONPATH=src/GridKIT:src python -m GridKIT.scripts.train_run --run-dir runs/<id>
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def _setup_paths() -> None:
    # Two import styles coexist in this codebase: core/grid_model/scenarios
    # import each other bare ("core.X"), rl_engine/scripts import "GridKIT.X"
    # — both roots must be on the path for either style to resolve.
    script_dir = Path(__file__).resolve().parent.parent      # src/GridKIT/
    src_dir = script_dir.parent                               # src/
    for p in (str(src_dir), str(script_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["PYTHONPATH"] = f"{script_dir}:{src_dir}"
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"


# ══════════════════════════════════════════════════════════════
# household_configuration.json (map_ui/household_config.py's schema) ->
# core.models.HouseholdDevices. Pure + import-light, so it's unit-testable
# without pulling in Ray/RLlib.
# ══════════════════════════════════════════════════════════════
def household_devices_from_config(config: dict, household_bus_ids: list[str]) -> dict:
    """Convert a map_ui household_configuration.json dict into a device layout.

    Known gaps in that schema as of map_ui/household_config.py — not bugs in
    this converter, just not there yet on the map_ui side:
      - no battery / PV concept at all -> every household comes back with
        battery=False, pv=False regardless of what the UI shows.
      - "load_scaling_by_bus" (a per-household consumption multiplier) has no
        home in HouseholdDevices/GridEnv yet (only a per-EPISODE global
        multiplier exists — core.constants.LOAD_MULTIPLIER_MIN/MAX) — read
        here for nothing else, deliberately not applied.
    """
    from core.models import HouseholdDevices

    resolved = config.get("resolved", {})
    ev_ids = set(resolved.get("ev_bus_ids", []))
    hp_ids = set(resolved.get("heat_pump_bus_ids", []))
    return {
        bus_id: HouseholdDevices(bus_id=bus_id, ev=bus_id in ev_ids, heat_pump=bus_id in hp_ids)
        for bus_id in household_bus_ids
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one run saved via core.run_store")
    parser.add_argument("--run-dir", required=True, help="runs/<id> directory")
    args = parser.parse_args()

    _setup_paths()
    import warnings
    warnings.filterwarnings("ignore")

    run_path = Path(args.run_dir).resolve()
    run_id = run_path.name
    root = str(run_path.parent)

    from core import run_store as rs

    try:
        import core.constants as const
        from grid_model.builder import FixedNetworkBuilder
        from scripts.run_experiment import _timeline
        from GridKIT.grid_model.environment import GridEnv
        from GridKIT.rl_engine import (
            GridEnvRLlibWrapper, Trainer, create_ippo_config, RLlibPolicyAdapter,
        )
        from GridKIT.scenarios.policies import NaiveImmediatePolicy, NaivePriceFollowPolicy
        from GridKIT.scenarios.runner import run_episode, run_scenario

        network = rs.load_network(run_id, root=root)
        household_configuration = rs.load_household_configuration(run_id, root=root) or {}
        layout = household_devices_from_config(household_configuration, list(network.household_bus_ids))

        iterations = const.PIPELINE_TRAINING_ITERATIONS
        seeds = list(range(const.PIPELINE_EVALUATION_SEEDS))

        active_device_types = tuple(sorted({d for cfg in layout.values() for d in cfg.controllable}))
        if not active_device_types:
            rs.set_status(run_id, state=rs.FAILED,
                           message="No EV/battery/heat-pump households in this configuration — nothing to train.",
                           root=root)
            return

        rs.set_status(run_id, state=rs.RUNNING, progress=0.0, iteration=0, total_iters=iterations,
                       message="starting Ray + training", root=root)

        # ── train on this exact grid ─────────────────────────
        def env_factory(cfg=None):
            return GridEnvRLlibWrapper(env=GridEnv(builder=FixedNetworkBuilder(network), device_layout=layout))

        def config_func(env_name):
            return create_ippo_config(env_name=env_name, device_types=active_device_types)

        class _StatusCallback:
            def on_iteration_end(self, iteration, reward, length):
                rs.set_status(run_id, state=rs.RUNNING, iteration=iteration, total_iters=iterations,
                               progress=iteration / max(1, iterations),
                               message=f"iter {iteration}/{iterations} · reward {reward:.1f}", root=root)

        trainer = Trainer(env_factory=env_factory, config_func=config_func)
        trainer.run(num_episodes=iterations, callback=_StatusCallback(), cleanup=False)

        rs.set_status(run_id, state=rs.RUNNING, progress=1.0, iteration=iterations,
                       total_iters=iterations, message="evaluating scenarios", root=root)
        trainer.save_checkpoint(str((run_path / "checkpoints").resolve()))
        modules = {dev: trainer.get_policy_module(f"{dev}_policy") for dev in active_device_types}
        adapter = RLlibPolicyAdapter(modules)

        # ── evaluate baselines + RL on the same grid ─────────
        scenarios = {
            "1: flat / immediate": NaiveImmediatePolicy(),
            "2: price-follow (automated)": NaivePriceFollowPolicy(jitter_std=0.0),
            "3: selfish RL": adapter,
        }
        summary: list[dict] = []
        timelines: dict[str, dict] = {}
        for label, policy in scenarios.items():
            env = GridEnv(builder=FixedNetworkBuilder(network), device_layout=layout)
            stats = run_scenario(env, policy, seeds, label=label)
            summary.append({
                "scenario": label,
                "curtailment_mean": stats.curtailment_events[0], "curtailment_std": stats.curtailment_events[1],
                "soc_mean": stats.soc_satisfaction_rate[0], "soc_std": stats.soc_satisfaction_rate[1],
                "peak_mean": stats.transformer_peak_loading_pu[0], "peak_std": stats.transformer_peak_loading_pu[1],
                "reward_mean": stats.mean_episode_reward[0], "reward_std": stats.mean_episode_reward[1],
                "bill_mean": stats.mean_household_bill_eur[0], "bill_std": stats.mean_household_bill_eur[1],
            })
            env = GridEnv(builder=FixedNetworkBuilder(network), device_layout=layout)
            rep = run_episode(env, policy, seeds[0])
            timelines[label] = _timeline(env, rep, label, 0.0)

        rs.save_results(run_id, summary, timelines, root=root)
        trainer.stop()

    except BaseException as exc:  # noqa: BLE001 — including KeyboardInterrupt/SystemExit: this
        # process's only job is to train and report status, so however it's ending, record FAILED
        # before we go. (A SIGKILL — e.g. an OOM kill — can't be caught by anything, by any process,
        # ever; that case is invisible here and only shows up as a stale run — see run_store.STALE_AFTER_SECONDS.)
        rs.set_status(run_id, state=rs.FAILED, message=f"{type(exc).__name__}: {exc}", root=root)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
