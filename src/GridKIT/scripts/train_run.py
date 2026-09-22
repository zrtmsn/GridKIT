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
# (core.constants.PIPELINE_MAX_TRAINING_ITERATIONS/PIPELINE_MIN_TRAINING_ITERATIONS/
# PIPELINE_EARLY_STOP_PATIENCE, PIPELINE_EVALUATION_SEEDS), not read from the run
# config — the UI never exposes RL knobs to the user, so there is nothing
# per-run to read here.
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
    # os.pathsep, not ":" — Ray starts worker processes that re-read PYTHONPATH,
    # and on Windows a colon-joined value leaves them unable to import GridKIT at
    # all. It surfaces far from here, as an IndexError inside the trainer.
    os.environ["PYTHONPATH"] = f"{script_dir}{os.pathsep}{src_dir}"
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"


# ══════════════════════════════════════════════════════════════
# household_configuration.json (map_ui/household_config.py's schema) ->
# core.models.HouseholdDevices. Pure + import-light, so it's unit-testable
# without pulling in Ray/RLlib.
# ══════════════════════════════════════════════════════════════
def household_devices_from_config(config: dict, household_bus_ids: list[str]) -> dict:
    """Convert a map_ui household_configuration.json dict into a device layout.

    ev/heat_pump/battery/pv ownership and pv_kwp/battery_kwh capacities come
    from `resolved.*` — either GridCreator's real per-household assignment or
    the scenario-assumption fallback, whichever map_ui/household_config.py
    resolved for this network (see its `build_household_configuration`).

    Known gap in that schema as of map_ui/household_config.py — not a bug in
    this converter, just not there yet on the map_ui side:
      - "load_scaling_by_bus" (a per-household consumption multiplier) has no
        home in HouseholdDevices/GridEnv yet (only a per-EPISODE global
        multiplier exists — core.constants.LOAD_MULTIPLIER_MIN/MAX) — read
        here for nothing else, deliberately not applied.
    """
    from core.models import HouseholdDevices

    resolved = config.get("resolved", {})
    ev_ids = set(resolved.get("ev_bus_ids", []))
    hp_ids = set(resolved.get("heat_pump_bus_ids", []))
    battery_ids = set(resolved.get("battery_bus_ids", []))
    pv_ids = set(resolved.get("pv_bus_ids", []))
    pv_kwp_by_bus = resolved.get("pv_kwp_by_bus", {})
    battery_kwh_by_bus = resolved.get("battery_kwh_by_bus", {})
    return {
        bus_id: HouseholdDevices(
            bus_id=bus_id, ev=bus_id in ev_ids, heat_pump=bus_id in hp_ids,
            battery=bus_id in battery_ids, battery_kwh=battery_kwh_by_bus.get(bus_id),
            pv=bus_id in pv_ids, pv_kwp=pv_kwp_by_bus.get(bus_id),
        )
        for bus_id in household_bus_ids
    }


def configured_share(layout: dict) -> float:
    """Share of households that got at least one controllable device.

    The dashboard labels its results with an "Ausstattungsgrad". In the batch
    experiment that is the swept input; a run from the map has no sweep, so it
    is read back off the layout the user actually configured. Households with
    only PV don't count — PV isn't controllable, so it is not part of what the
    §14a question is about.
    """
    if not layout:
        return 0.0
    return sum(1 for cfg in layout.values() if cfg.controllable) / len(layout)


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
        from scripts.run_experiment import _timeline, summary_record
        from GridKIT.grid_model.environment import GridEnv
        from GridKIT.rl_engine import (
            GridEnvRLlibWrapper, Trainer, create_ippo_config, RLlibPolicyAdapter,
        )
        from GridKIT.scenarios.policies import NaiveImmediatePolicy, NaivePriceFollowPolicy
        from GridKIT.scenarios.runner import run_episode, run_scenario

        network = rs.load_network(run_id, root=root)
        household_configuration = rs.load_household_configuration(run_id, root=root) or {}
        layout = household_devices_from_config(household_configuration, list(network.household_bus_ids))

        iterations = const.PIPELINE_MAX_TRAINING_ITERATIONS
        min_iterations = const.PIPELINE_MIN_TRAINING_ITERATIONS
        early_stop_patience = const.PIPELINE_EARLY_STOP_PATIENCE
        seeds = list(range(const.PIPELINE_EVALUATION_SEEDS))

        # What the dashboard shows as the Ausstattungsgrad for this run.
        PENETRATION = configured_share(layout)
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

        def config_func(env_name, *, train_batch_size=None, minibatch_size=None, num_env_runners=None):
            # Explicitly accept the agent-count-scaled values: Trainer detects
            # them via inspect.signature and re-sizes the batch to the user's
            # network size, so the UI pipeline benefits from the same scaling as
            # the batch experiment. None values fall back to create_ippo_config's
            # configured defaults.
            kwargs = {}
            if train_batch_size is not None:
                kwargs["train_batch_size"] = train_batch_size
            if minibatch_size is not None:
                kwargs["minibatch_size"] = minibatch_size
            if num_env_runners is not None:
                kwargs["num_env_runners"] = num_env_runners
            return create_ippo_config(env_name=env_name, device_types=active_device_types, **kwargs)

        class _StatusCallback:
            def on_iteration_end(self, iteration, reward, length):
                rs.set_status(run_id, state=rs.RUNNING, iteration=iteration, total_iters=iterations,
                               progress=iteration / max(1, iterations),
                               message=f"iter {iteration}/{iterations} · reward {reward:.1f}", root=root)

        trainer = Trainer(env_factory=env_factory, config_func=config_func)
        # metrics_dir: without it iteration_metrics.json goes to a temp path and
        # the dashboard's Training tab stays empty for every run from the map.
        # num_episodes is only the ceiling: the run stops earlier once the
        # reward has plateaued (convergence.py), so a user's network size never
        # dictates a fixed iteration count.
        results = trainer.run(num_episodes=iterations, callback=_StatusCallback(), cleanup=False,
                              metrics_dir=run_path,
                              min_iterations=min_iterations, patience=early_stop_patience)

        # Report WHY training ended (reward-converged vs full budget) to the UI.
        if results and results[-1].early_stopped:
            converged_note = f" · konvergiert nach {len(results)}/{iterations} Iterationen"
        else:
            converged_note = ""
        rs.set_status(run_id, state=rs.RUNNING, progress=1.0, iteration=iterations,
                       total_iters=iterations, message=f"evaluating scenarios{converged_note}", root=root)
        trainer.save_checkpoint(str((run_path / "checkpoints").resolve()))
        modules = {dev: trainer.get_policy_module(f"{dev}_policy") for dev in active_device_types}
        adapter = RLlibPolicyAdapter(modules)

        # ── evaluate baselines + RL on the same grid ─────────
        # The same four scenarios as the batch experiment, so a run started from
        # the map is directly comparable with one from run_experiment.py — the
        # Szenarienvergleich tab is built around this set.
        scenarios = {
            "1: flat / immediate": NaiveImmediatePolicy(),
            "2: price-follow (manual)": NaivePriceFollowPolicy(jitter_std=8.0),
            "2: price-follow (automated)": NaivePriceFollowPolicy(jitter_std=0.0),
            "3: selfish RL": adapter,
        }
        summary: list[dict] = []
        # A list, not a dict keyed by label: the dashboard's tabs filter timelines
        # by their `scenario` field and read them as records, the same shape
        # run_experiment.py writes. A dict here left every tab but the first empty.
        timelines: list[dict] = []
        for label, policy in scenarios.items():
            env = GridEnv(builder=FixedNetworkBuilder(network), device_layout=layout)
            stats = run_scenario(env, policy, seeds, label=label)
            # summary_record, not an inline dict: the dashboard reads ~30 fields
            # (HP comfort, battery cycles, the feeder/line breakdown) that an
            # inline subset leaves blank.
            summary.append(summary_record(stats, label, PENETRATION))
            env = GridEnv(builder=FixedNetworkBuilder(network), device_layout=layout)
            rep = run_episode(env, policy, seeds[0])
            timelines.append(_timeline(env, rep, label, PENETRATION))

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
