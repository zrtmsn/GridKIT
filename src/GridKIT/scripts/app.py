# scripts/app.py
# ─────────────────────────────────────────────────────────────
# GridKIT end-to-end entry point: (optionally) run the full
# scenario × penetration experiment, then open the dashboard.
#
#   gridkit-app                      # open dashboard on existing results
#   gridkit-app --train              # run the experiment first, then open
#   gridkit-app --train --network data/karlsruhe.json
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _paths():
    script_dir = Path(__file__).resolve().parent.parent   # src/GridKIT/
    src_dir = script_dir.parent                            # src/
    root = src_dir.parent
    return script_dir, src_dir, root


def main() -> None:
    parser = argparse.ArgumentParser(description="GridKIT — run experiment and/or open dashboard")
    parser.add_argument("--train", action="store_true", help="run the experiment before opening the dashboard")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--network", type=str, default="data/feeder_20.json")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--no-dashboard", action="store_true", help="run the experiment only, skip the UI")
    args = parser.parse_args()

    script_dir, src_dir, root = _paths()
    os.chdir(root)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{script_dir}{os.pathsep}{src_dir}"
    env["GRIDKIT_OUTPUT_DIR"] = args.out

    summary = Path(args.out) / "summary.json"
    if args.train or not summary.exists():
        if not summary.exists() and not args.train:
            print(f"No results at {summary} — running the experiment first.")
        subprocess.run(
            [sys.executable, "-m", "GridKIT.scripts.run_experiment",
             "--iterations", str(args.iterations), "--seeds", str(args.seeds),
             "--network", args.network, "--out", args.out],
            env=env, check=True,
        )

    if args.no_dashboard:
        return

    dashboard = script_dir / "dashboard" / "app.py"
    print(f"Launching dashboard: streamlit run {dashboard}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard)],
        env=env, check=False,
    )


if __name__ == "__main__":
    main()
