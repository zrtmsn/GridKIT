# scripts/train_run.py
# ─────────────────────────────────────────────────────────────
# Background trainer for ONE designed grid + device layout (a "run").
#
# Reads a run directory (created by run_store.create_run), trains an IPPO policy
# on that exact grid, evaluates the naive baselines + the trained RL, and writes
# summary.json / timelines.json / checkpoints into the run dir — updating
# status.json live so the web app can show progress while it trains.
#
# Launched detached by the web app:
#   PYTHONPATH=src python -m GridKIT.scripts.train_run --run-dir runs/<id>
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _setup_paths() -> None:
    script_dir = Path(__file__).resolve().parent.parent      # src/GridKIT/
    src_dir = script_dir.parent                              # src/
    for p in (str(src_dir), str(script_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["PYTHONPATH"] = f"{script_dir}{os.pathsep}{src_dir}"
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one designed grid (a run)")
    parser.add_argument("--run-dir", required=True, help="runs/<id> directory")
    args = parser.parse_args()

    _setup_paths()
    import warnings
    warnings.filterwarnings("ignore")

    run_path = Path(args.run_dir).resolve()
    run_id = run_path.name
    root = str(run_path.parent)

    from scripts import run_store as rs

    try:
        import core.constants as const
        from core.models import GridNetwork  # noqa: F401
        from scripts.grid_designer import StaticBuilder
        from scripts.run_experiment import _timeline
        from GridKIT.grid_model.environment import GridEnv
        from GridKIT.grid_model.device_profiles import DeviceProfileProvider
        from GridKIT.rl_engine import (
            GridEnvRLlibWrapper, Trainer, create_ippo_config, RLlibPolicyAdapter,
        )
        from GridKIT.scenarios.policies import NaiveImmediatePolicy, NaivePriceFollowPolicy
        from GridKIT.scenarios.runner import run_episode, run_scenario

        config = rs.load_config(run_id, root=root) or {}
        iterations = int(config.get("iterations", 20))
        seeds = list(range(int(config.get("seeds", 6))))
        network = rs.load_network(run_id, root=root)
        layout = rs.load_layout(run_id, root=root)
        n_house = len(network.household_bus_ids)
        n_active = sum(1 for c in layout.values() if c.controllable)
        pen_label = round(n_active / n_house, 2) if n_house else 0.0

        rs.set_status(run_id, state=rs.RUNNING, progress=0.0, iteration=0,
                      total_iters=iterations, message="starting Ray + training", root=root)

        # ── train on this exact grid ─────────────────────────
        def env_factory(cfg=None):
            return GridEnvRLlibWrapper(env=GridEnv(builder=StaticBuilder(network), device_layout=layout))

        class _StatusCallback:
            def on_iteration_end(self, iteration, reward, length):
                rs.set_status(run_id, state=rs.RUNNING, iteration=iteration, total_iters=iterations,
                              progress=iteration / max(1, iterations),
                              message=f"iter {iteration}/{iterations} · reward {reward:.1f}", root=root)

        trainer = Trainer(env_factory=env_factory, config_func=create_ippo_config)
        trainer.run(num_episodes=iterations, callback=_StatusCallback(), cleanup=False,
                    metrics_dir=run_path)

        rs.set_status(run_id, state=rs.RUNNING, progress=1.0, iteration=iterations,
                      total_iters=iterations, message="evaluating scenarios", root=root)
        trainer.save_checkpoint(str((run_path / "checkpoints").resolve()))
        adapter = RLlibPolicyAdapter(trainer.get_policy_modules())

        # ── evaluate baselines + RL on the same grid ─────────
        provider = DeviceProfileProvider.build()
        scenarios = {
            "1: flat / immediate": NaiveImmediatePolicy(),
            "2: price-follow (manual)": NaivePriceFollowPolicy(jitter_std=8.0),
            "2: price-follow (automated)": NaivePriceFollowPolicy(jitter_std=0.0),
            "3: selfish RL": adapter,
        }
        summary: list[dict] = []
        timelines: list[dict] = []
        for label, policy in scenarios.items():
            env = GridEnv(builder=StaticBuilder(network), device_layout=layout, profile_provider=provider)
            stats = run_scenario(env, policy, seeds, label=label, ev_penetration=pen_label)
            # line peak is reported alongside the transformer peak because in a real LV
            # grid the cable binds first: Oberacker's transformers sit near 0.6 pu while
            # feeder-head cables hit 1.09 pu, so the transformer figure alone reads
            # "comfortable" through an actual thermal violation.
            line_peaks = [v[0] for v in stats.line_peak_loading_pu.values()]
            worst = stats.worst_feeder()
            summary.append({
                "penetration": pen_label, "scenario": label,
                "curtailment_mean": stats.curtailment_events[0], "curtailment_std": stats.curtailment_events[1],
                "soc_mean": stats.soc_satisfaction_rate[0], "soc_std": stats.soc_satisfaction_rate[1],
                "peak_mean": stats.transformer_peak_loading_pu[0], "peak_std": stats.transformer_peak_loading_pu[1],
                "reward_mean": stats.mean_episode_reward[0], "reward_std": stats.mean_episode_reward[1],
                # the adoption test: a grid-friendly policy nobody would install is worthless
                "bill_mean": stats.mean_household_bill_eur[0], "bill_std": stats.mean_household_bill_eur[1],
                # where the stress actually was
                "line_peak_max": max(line_peaks, default=0.0),
                "n_lines_overloaded": len(stats.line_overload_steps),
                "worst_feeder": worst[0] if worst else None,
                "worst_feeder_steps": worst[1] if worst else 0.0,
                "feeder_overload_steps": {k: v[0] for k, v in stats.feeder_overload_steps.items()},
                "feeder_peak_loading_pu": {k: v[0] for k, v in stats.feeder_peak_loading_pu.items()},
                "line_overload_steps": {k: v[0] for k, v in stats.line_overload_steps.items()},
                "line_peak_loading_pu": {k: v[0] for k, v in stats.line_peak_loading_pu.items()},
            })
            env = GridEnv(builder=StaticBuilder(network), device_layout=layout, profile_provider=provider)
            rep = run_episode(env, policy, seeds[0], ev_penetration=pen_label)
            timelines.append(_timeline(env, rep, label, pen_label))

        (run_path / "summary.json").write_text(json.dumps(summary, indent=2))
        (run_path / "timelines.json").write_text(json.dumps(timelines, indent=2))
        trainer.stop()
        rs.set_status(run_id, state=rs.DONE, progress=1.0, iteration=iterations,
                      total_iters=iterations, message="done", root=root)

    except Exception as exc:  # noqa: BLE001
        rs.set_status(run_id, state=rs.FAILED, message=f"{type(exc).__name__}: {exc}", root=root)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
