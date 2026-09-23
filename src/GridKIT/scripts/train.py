# scripts/train.py
# ─────────────────────────────────────────────────────────────
# Scenario 3 — selfish, congestion-aware RL (shared-policy IPPO).
# Headless training entry point. Wires GridEnv → GridEnvRLlibWrapper →
# Trainer(create_ippo_config) and runs a (by default short) training loop.
#
# Usage:  python -m GridKIT.scripts.train [--max-iterations N]
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _setup_paths() -> None:
    """Put src/ and src/GridKIT/ on the path for the driver *and* Ray workers."""
    script_dir = Path(__file__).resolve().parent.parent   # src/GridKIT/
    src_dir = script_dir.parent                            # src/
    for p in (str(src_dir), str(script_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(src_dir.parent)                               # project root (for stub_network.json)
    os.environ["PYTHONPATH"] = f"{script_dir}{os.pathsep}{src_dir}"
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"


def main() -> None:
    parser = argparse.ArgumentParser(description="GridKIT scenario-3 IPPO training")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="training iterations (ceiling; small default = smoke run, not convergence)")
    args = parser.parse_args()

    _setup_paths()

    from GridKIT.grid_model.environment import GridEnv
    from GridKIT.rl_engine import GridEnvRLlibWrapper, Trainer, create_ippo_config

    def env_factory(config=None):
        # fresh Env + Wrapper per Ray worker; OBS_DIM=7 obs space is read from constants
        return GridEnvRLlibWrapper(env=GridEnv())

    trainer = Trainer(env_factory=env_factory, config_func=create_ippo_config)
    results = trainer.run(num_episodes=args.max_iterations)

    print(f"\nTraining finished — {len(results)} iterations.")
    for r in results:
        print(f"  iter {r.iteration:>3}: return_mean={r.episode_return_mean:8.3f}  "
              f"len_mean={r.episode_len_mean:6.1f}")


if __name__ == "__main__":
    main()
