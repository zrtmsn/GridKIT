# scripts/run_scenarios.py
# ─────────────────────────────────────────────────────────────
# Composition root for the non-RL scenario comparison (scenarios 1 & 2).
# Wires StubNetworkBuilder → GridEnv → policy → runner and prints a
# curtailment table over seeds, including the σ-sweep that shows
# synchronization (and thus §14a curtailment) emerging as the naive
# price-follow behaviour becomes more automated (σ → 0).
#
# Usage:  python -m GridKIT.scripts.run_scenarios [--seeds N]
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import warnings

from GridKIT.grid_model.environment import GridEnv
from GridKIT.scenarios.policies import NaiveImmediatePolicy, NaivePriceFollowPolicy
from GridKIT.scenarios.runner import run_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="GridKIT non-RL scenario comparison")
    parser.add_argument("--seeds", type=int, default=6, help="number of seeded episodes per scenario")
    parser.add_argument("--scenario", default="medium", choices=["low", "medium", "high"],
                        help="price scenario for the dynamic-pricing cases")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    seeds = list(range(args.seeds))
    env = GridEnv(price_scenario=args.scenario)

    print(f"\nGridKIT scenario comparison — {args.seeds} seeds, price={args.scenario}, "
          f"{env.network.n_households} households\n" + "─" * 100)

    # Scenario 1 — flat tariff, charge immediately on arrival
    stats = [run_scenario(env, NaiveImmediatePolicy(), seeds, label="1: flat / immediate")]

    # Scenario 2 — naive price-follow, swept from manual (wide σ) to automated (σ→0)
    for sigma, tag in [(8.0, "manual σ=8"), (4.0, "σ=4"), (1.0, "σ=1"), (0.0, "automated σ=0")]:
        policy = NaivePriceFollowPolicy(jitter_std=sigma)
        stats.append(run_scenario(env, policy, seeds, label=f"2: price-follow {tag}"))

    print("\n".join(str(s) for s in stats))
    print("─" * 100)
    print("Expectation: curtailment rises as σ→0 (everyone piles into the same cheap overnight "
          "window).\nNote: metrics are demonstrated-mechanism comparisons under simplified inputs, "
          "not real-grid forecasts.\n")


if __name__ == "__main__":
    main()
