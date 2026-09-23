# scripts/run_experiment.py
# ─────────────────────────────────────────────────────────────
# Full GridKIT experiment: for each EV penetration level, train the
# selfish congestion-aware IPPO policy (scenario 3), then evaluate
# scenarios 1–3 over seeds and save results for plotting.
#
# Output (under --out, default outputs/):
#   summary.json    — per (penetration, scenario) curtailment / SoC / peak (mean±std)
#   timelines.json  — a representative 24 h episode per (penetration, scenario)
#   checkpoints/    — trained policy per penetration
#
# Usage:  python -m GridKIT.scripts.run_experiment [--max-iterations N] [--seeds M]
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np


def _setup_paths() -> Path:
    import sys
    script_dir = Path(__file__).resolve().parent.parent      # src/GridKIT/
    src_dir = script_dir.parent                              # src/
    for p in (str(src_dir), str(script_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    root = src_dir.parent
    os.chdir(root)
    os.environ["PYTHONPATH"] = f"{script_dir}{os.pathsep}{src_dir}"
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    return root


def summary_record(stats, label: str, penetration: float) -> dict:
    """One row of summary.json, from an aggregated ScenarioStats.

    Shared by both producers on purpose. run_experiment.py (batch sweep) and
    train_run.py (the web app's background training) each used to build this
    dict inline, and the two had drifted into different schemas: the batch path
    wrote HP comfort and the battery figures, the web-app path wrote the
    feeder/line breakdown, and neither was a superset. A dashboard tile fed by
    either group was therefore empty for exactly half the runs, depending on
    how they had been started. This returns the union, so both paths write the
    same fields.
    """
    import core.constants as const

    charge_mean, charge_std = stats.battery_charge_kwh
    discharge_mean, discharge_std = stats.battery_discharge_kwh
    throughput_mean = charge_mean + discharge_mean
    # charge and discharge are aggregated separately, so their spreads combine
    # in quadrature rather than adding
    throughput_std = float(np.sqrt(charge_std ** 2 + discharge_std ** 2))
    line_peaks = [v[0] for v in stats.line_peak_loading_pu.values()]
    worst = stats.worst_feeder()

    return {
        "penetration": penetration,
        "scenario": label,
        "curtailment_mean": stats.curtailment_events[0], "curtailment_std": stats.curtailment_events[1],
        "soc_mean": stats.soc_satisfaction_rate[0], "soc_std": stats.soc_satisfaction_rate[1],
        "peak_mean": stats.transformer_peak_loading_pu[0], "peak_std": stats.transformer_peak_loading_pu[1],
        "reward_mean": stats.mean_episode_reward[0], "reward_std": stats.mean_episode_reward[1],
        # the adoption test: a grid-friendly policy nobody would install is worthless
        "bill_mean": stats.mean_household_bill_eur[0], "bill_std": stats.mean_household_bill_eur[1],
        # without these the scenario comparison is silently EV-only, even though
        # heat pump and battery run in every episode
        "hp_comfort_mean": stats.hp_comfort_satisfaction_rate[0],
        "hp_comfort_std": stats.hp_comfort_satisfaction_rate[1],
        "battery_charge_kwh_mean": charge_mean, "battery_charge_kwh_std": charge_std,
        "battery_discharge_kwh_mean": discharge_mean, "battery_discharge_kwh_std": discharge_std,
        "battery_throughput_kwh_mean": throughput_mean, "battery_throughput_kwh_std": throughput_std,
        # Full Equivalent Cycles: throughput over both legs against one full capacity
        "battery_full_cycles_mean": throughput_mean / 2.0 / const.BATTERY_CAPACITY_KWH,
        "battery_full_cycles_std": throughput_std / 2.0 / const.BATTERY_CAPACITY_KWH,
        # where the stress actually was — the cable usually binds before the transformer
        "line_peak_max": max(line_peaks, default=0.0),
        "n_lines_overloaded": len(stats.line_overload_steps),
        "worst_feeder": worst[0] if worst else None,
        "worst_feeder_steps": worst[1] if worst else 0.0,
        "feeder_overload_steps": {k: v[0] for k, v in stats.feeder_overload_steps.items()},
        "feeder_peak_loading_pu": {k: v[0] for k, v in stats.feeder_peak_loading_pu.items()},
        "line_overload_steps": {k: v[0] for k, v in stats.line_overload_steps.items()},
        "line_peak_loading_pu": {k: v[0] for k, v in stats.line_peak_loading_pu.items()},
    }


def _timeline(env, result, label: str, penetration: float) -> dict:
    # Imported here, not at module level: _setup_paths() puts the source roots on
    # sys.path at call time, so a top-level import would run too early. (The
    # cumulative-battery lines below used `const` without this and always raised
    # NameError — main()'s own import is a local, not visible in this function.)
    import core.constants as const

    tr = result.timestep_results
    return {
        "penetration": penetration,
        "scenario": label,
        "transformer_loading": [pf.transformer_loading_pu for pf in tr],
        "max_line_loading": [max(pf.line_loadings_pu.values()) for pf in tr],
        "curtailment": [bool(pf.curtailment_applied) for pf in tr],
        
        # ════════════════════════════════════════════════════════════════
        # NEW: IDs of the overloaded network elements (per timestep)
        # ════════════════════════════════════════════════════════════════
        # Only logged in steps with curtailment_applied=True (grid control active)
        # Format: list of lists [[], ["line_42"], ["trafo_1", "line_73"], ...]
        "overloaded_transformers": [
            [tid for tid, loading in pf.transformer_loadings_pu.items() if loading > 1.0]
            for pf in tr
        ],
        "overloaded_lines": [
            [lid for lid, loading in pf.line_loadings_pu.items() if loading > 1.0]
            for pf in tr
        ],
        
        "price": env.day_ahead_prices(),
        "base_load": env.episode_base_load_kw,
        "pv_generation": env.episode_pv_kw,
        "temperature": env.episode_temperature_c,
        # per-device decisions (feeder-aggregate delivered power): what each device type chose
        "ev_power": [pf.device_power_kw.get("ev", 0.0) for pf in tr],
        "battery_power": [pf.device_power_kw.get("battery", 0.0) for pf in tr],   # signed: + charge / − discharge
        "hp_power": [pf.device_power_kw.get("hp", 0.0) for pf in tr],
        # ONE representative household — exact device power + EV availability + SoC traces
        "house_ev_power": [pf.sample_household.get("ev_kw", 0.0) for pf in tr],
        "house_battery_power": [pf.sample_household.get("battery_kw", 0.0) for pf in tr],
        "house_hp_power": [pf.sample_household.get("hp_kw", 0.0) for pf in tr],
        "house_pv": [pf.sample_household.get("pv_kw", 0.0) for pf in tr],
        "house_ev_available": [pf.sample_household.get("ev_available", 0.0) for pf in tr],
        "house_ev_soc": [pf.sample_household.get("ev_soc", 0.0) for pf in tr],
        "house_battery_soc": [pf.sample_household.get("battery_soc", 0.0) for pf in tr],
        "house_hp_soc": [pf.sample_household.get("hp_soc", 0.0) for pf in tr],
        # Battery cycle counting (cumulative energy for Full Equivalent Cycles calculation)
        "house_battery_charge_cumulative_kwh": np.cumsum(
            [max(0.0, pf.sample_household.get("battery_kw", 0.0)) * const.TIMESTEP_HOURS for pf in tr]
        ).tolist(),
        "house_battery_discharge_cumulative_kwh": np.cumsum(
            [max(0.0, -pf.sample_household.get("battery_kw", 0.0)) * const.TIMESTEP_HOURS for pf in tr]
        ).tolist(),
        "soc_satisfaction_rate": result.metrics.soc_satisfaction_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GridKIT full scenario × penetration experiment")
    parser.add_argument("--max-iterations", type=int, default=40,
                        help="MAX IPPO training iterations per penetration (hard ceiling; "
                             "training stops earlier once the reward has converged)")
    parser.add_argument("--min-iterations", type=int, default=5,
                        help="never early-stop before this many iterations (default 5)")
    parser.add_argument("--patience", type=int, default=6,
                        help="stop after this many iterations without reward improvement (default 6)")
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
        print(f"\n=== EV penetration {pen:.0%} — training scenario 3 (≤{args.max_iterations} iters, early stop) ===")

        def env_factory(cfg=None, _pen=pen):
            return GridEnvRLlibWrapper(env=GridEnv(ev_penetration=_pen, builder=StubNetworkBuilder(path=network_path)))

        checkpoint_dir = out / "checkpoints" / f"pen_{int(pen * 100)}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(env_factory=env_factory, config_func=create_ippo_config)
        # Write iteration_metrics.json straight into this penetration's checkpoint dir.
        # It used to go to a temp file and be copied back from a hardcoded "/tmp/..."
        # path, which does not exist on Windows — so the copy silently did nothing and
        # the dashboard's training tab never found any metrics.
        trainer.run(num_episodes=args.max_iterations, cleanup=False, metrics_dir=checkpoint_dir,
                    min_iterations=args.min_iterations, patience=args.patience)

        trainer.save_checkpoint(str(checkpoint_dir.resolve()))
        print(f"  Saved iteration metrics → {checkpoint_dir / 'iteration_metrics.json'}")


        adapter = RLlibPolicyAdapter(trainer.get_policy_modules())

        eval_env = GridEnv(ev_penetration=pen, builder=StubNetworkBuilder(path=network_path))
        scenarios = {
            "1: flat / immediate": NaiveImmediatePolicy(),
            "2: price-follow (manual)": NaivePriceFollowPolicy(jitter_std=8.0),
            "2: price-follow (automated)": NaivePriceFollowPolicy(jitter_std=0.0),
            "3: selfish RL": adapter,
        }
        for label, policy in scenarios.items():
            # ════════════════════════════════════════════════════════════════
            # LOGGING STAGE 1: evaluate over multiple seeds (default: 12)
            # ════════════════════════════════════════════════════════════════
            # run_scenario() runs 12 episodes and aggregates the metrics into
            # mean±std (curtailment, SoC, peak loading, reward, bill, etc.)
            stats = run_scenario(eval_env, policy, seeds, label=label, ev_penetration=pen)
            
            # ════════════════════════════════════════════════════════════════
            # LOGGING STAGE 2: append the aggregated metrics to the summary list
            # ════════════════════════════════════════════════════════════════
            # These data later end up in summary.json (for bar charts)
            summary.append(summary_record(stats, label, pen))
            print("  " + str(stats))
            
            # ════════════════════════════════════════════════════════════════
            # LOGGING STAGE 3: store detailed time series for ONE episode
            # ════════════════════════════════════════════════════════════════
            # run_episode() with seed[0] (the first of the 12 episodes) returns
            # complete 96-step time series (transformer loading, device power, SoC, etc.)
            # These data later end up in timelines.json (for line charts)
            rep = run_episode(eval_env, policy, seeds[0], ev_penetration=pen)
            timelines.append(_timeline(eval_env, rep, label, pen))

        trainer.stop()

    # ════════════════════════════════════════════════════════════════════════
    # LOGGING STAGE 4: write the JSON files (summary + timelines)
    # ════════════════════════════════════════════════════════════════════════
    # These two files are read by plot_results.py to generate PNGs
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "timelines.json").write_text(json.dumps(timelines, indent=2))
    print(f"\nSaved {len(summary)} scenario results → {out/'summary.json'} and timelines → {out/'timelines.json'}")


if __name__ == "__main__":
    main()
