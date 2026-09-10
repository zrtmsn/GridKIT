# scripts/regen_timelines.py
# ─────────────────────────────────────────────────────────────
# Regenerate outputs/timelines.json from ALREADY-TRAINED checkpoints
# (no retraining), e.g. after adding new per-step fields to the timeline
# record (device decisions, PV, temperature). Keeps summary.json untouched.
#
# Usage:  python -m GridKIT.scripts.regen_timelines [--out outputs] [--network data/feeder_20.json]
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path


def _setup_paths() -> None:
    import sys
    script_dir = Path(__file__).resolve().parent.parent      # src/GridKIT/
    src_dir = script_dir.parent                              # src/
    for p in (str(src_dir), str(script_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(src_dir.parent)
    os.environ["PYTHONPATH"] = f"{script_dir}{os.pathsep}{src_dir}"


def _load_adapter(ckpt_dir: str, devices):
    """Restore the trained per-device RLModules directly (no Algorithm / env runners)."""
    from ray.rllib.core.rl_module.rl_module import RLModule
    from GridKIT.rl_engine import RLlibPolicyAdapter
    base = os.path.join(ckpt_dir, "learner_group", "learner", "rl_module")
    modules = {d: RLModule.from_checkpoint(os.path.abspath(os.path.join(base, f"{d}_policy")))
               for d in devices}
    return RLlibPolicyAdapter(modules)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate timelines.json from saved checkpoints")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--network", type=str, default="data/feeder_20.json")
    parser.add_argument("--seed", type=int, default=0, help="representative episode seed")
    args = parser.parse_args()

    _setup_paths()
    warnings.filterwarnings("ignore")

    import core.constants as const
    from GridKIT.grid_model.builder import StubNetworkBuilder
    from GridKIT.grid_model.environment import GridEnv
    from GridKIT.scenarios.policies import NaiveImmediatePolicy, NaivePriceFollowPolicy
    from GridKIT.scenarios.runner import run_episode
    from GridKIT.scripts.run_experiment import _timeline

    out = Path(args.out)
    timelines: list[dict] = []
    for pen in const.EV_PENETRATION_LEVELS:
        ckpt = out / "checkpoints" / f"pen_{int(pen * 100)}"
        adapter = _load_adapter(str(ckpt), const.CONTROLLABLE_DEVICE_TYPES)
        env = GridEnv(ev_penetration=pen, builder=StubNetworkBuilder(path=args.network))
        scenarios = {
            "1: flat / immediate": NaiveImmediatePolicy(),
            "2: price-follow (manual)": NaivePriceFollowPolicy(jitter_std=8.0),
            "2: price-follow (automated)": NaivePriceFollowPolicy(jitter_std=0.0),
            "3: selfish RL": adapter,
        }
        for label, policy in scenarios.items():
            rep = run_episode(env, policy, args.seed, ev_penetration=pen)
            timelines.append(_timeline(env, rep, label, pen))
        print(f"  regenerated timelines for penetration {pen:.0%}")

    (out / "timelines.json").write_text(json.dumps(timelines, indent=2))
    print(f"Wrote {len(timelines)} timelines (with device decisions) → {out/'timelines.json'} "
          f"(summary.json untouched)")


if __name__ == "__main__":
    main()
