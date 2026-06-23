# scripts/run_experiment.py
# ─────────────────────────────────────────────────────────────
# Full GridKIT experiment: for each EV penetration level, train the
# selfish congestion-aware IPPO policy (scenario 3), then evaluate
# scenarios 1–3 over seeds and save results for the dashboard.
#
# Output (under --out, default outputs/):
#   summary.json    — per (penetration, scenario) curtailment / SoC / peak (mean±std)
#   timelines.json  — a representative 24 h episode per (penetration, scenario)
#   checkpoints/    — trained policy per penetration
#
# Usage:  python -m GridKIT.scripts.train ... ; or
#         python -m GridKIT.scripts.run_experiment [--iterations N] [--seeds M]
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path


def _setup_paths() -> Path:
    import sys
    script_dir = Path(__file__).resolve().parent.parent      # src/GridKIT/
    src_dir = script_dir.parent                              # src/
    for p in (str(src_dir), str(script_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    root = src_dir.parent
    os.chdir(root)
    os.environ["PYTHONPATH"] = f"{script_dir}:{src_dir}"
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    return root


def _timeline(env, result, label: str, penetration: float) -> dict:
    tr = result.timestep_results
    return {
        "penetration": penetration,
        "scenario": label,
        "transformer_loading": [pf.transformer_loading_pu for pf in tr],
        "max_line_loading": [max(pf.line_loadings_pu.values()) for pf in tr],
        "curtailment": [bool(pf.curtailment_applied) for pf in tr],
        "price": env.day_ahead_prices(),
        "base_load": env.episode_base_load_kw,
        "soc_satisfaction_rate": result.metrics.soc_satisfaction_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GridKIT full scenario × penetration experiment")
    parser.add_argument("--iterations", type=int, default=40, help="IPPO training iterations per penetration")
    parser.add_argument("--seeds", type=int, default=12, help="evaluation episodes per scenario")
    parser.add_argument("--out", type=str, default="outputs", help="output directory")
    parser.add_argument("--network", type=str, default="data/feeder_20.json",
                        help="network JSON (default: 20-household feeder; use data/stub_network.json for the small stub)")
    args = parser.parse_args()

    _setup_paths()
    warnings.filterwarnings("ignore")

    import core.constants as const
    from GridKIT.grid_model.builder import StubNetworkBuilder
    from GridKIT.grid_model.environment import GridEnv
    from GridKIT.rl_engine import GridEnvRLlibWrapper, Trainer, create_ippo_config, RLlibPolicyAdapter
    from GridKIT.scenarios.policies import NaiveImmediatePolicy, NaivePriceFollowPolicy
    from GridKIT.scenarios.runner import run_episode, run_scenario

    network_path = args.network

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    penetrations = const.EV_PENETRATION_LEVELS

    summary: list[dict] = []
    timelines: list[dict] = []

    for pen in penetrations:
        print(f"\n=== EV penetration {pen:.0%} — training scenario 3 ({args.iterations} iters) ===")

        def env_factory(cfg=None, _pen=pen):
            return GridEnvRLlibWrapper(env=GridEnv(ev_penetration=_pen, builder=StubNetworkBuilder(path=network_path)))

        trainer = Trainer(env_factory=env_factory, config_func=create_ippo_config)
        trainer.run(num_episodes=args.iterations, cleanup=False)
        adapter = RLlibPolicyAdapter(trainer.get_policy_module())
        trainer.save_checkpoint(str((out / "checkpoints" / f"pen_{int(pen * 100)}").resolve()))

        eval_env = GridEnv(ev_penetration=pen, builder=StubNetworkBuilder(path=network_path))
        scenarios = {
            "1: flat / immediate": NaiveImmediatePolicy(),
            "2: price-follow (manual)": NaivePriceFollowPolicy(jitter_std=8.0),
            "2: price-follow (automated)": NaivePriceFollowPolicy(jitter_std=0.0),
            "3: selfish RL": adapter,
        }
        for label, policy in scenarios.items():
            stats = run_scenario(eval_env, policy, seeds, label=label, ev_penetration=pen)
            summary.append({
                "penetration": pen,
                "scenario": label,
                "curtailment_mean": stats.curtailment_events[0], "curtailment_std": stats.curtailment_events[1],
                "soc_mean": stats.soc_satisfaction_rate[0], "soc_std": stats.soc_satisfaction_rate[1],
                "peak_mean": stats.transformer_peak_loading_pu[0], "peak_std": stats.transformer_peak_loading_pu[1],
                "reward_mean": stats.mean_episode_reward[0], "reward_std": stats.mean_episode_reward[1],
            })
            print("  " + str(stats))
            rep = run_episode(eval_env, policy, seeds[0], ev_penetration=pen)
            timelines.append(_timeline(eval_env, rep, label, pen))

        trainer.stop()

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "timelines.json").write_text(json.dumps(timelines, indent=2))
    print(f"\nSaved {len(summary)} scenario results → {out/'summary.json'} and timelines → {out/'timelines.json'}")


if __name__ == "__main__":
    main()
