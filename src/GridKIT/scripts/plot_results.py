# scripts/plot_results.py
# ─────────────────────────────────────────────────────────────
# Headless (no Streamlit) plots for a run_experiment.py output directory —
# reads summary.json / timelines.json and saves PNGs. For a quick look at
# results without the dashboard UI.
#
# Usage: python -m GridKIT.scripts.plot_results --results outputs --out outputs/graphs
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical color order (dataviz skill palette, light-mode slots 1-4) —
# assigned by scenario identity, never re-cycled, so the same scenario always
# gets the same color across every chart in this run.
_SCENARIO_COLORS = {
    "1: flat / immediate": "#2a78d6",          # slot 1 blue
    "2: price-follow (manual)": "#eb6834",     # slot 2 orange
    "2: price-follow (automated)": "#1baf7a",  # slot 3 aqua
    "3: selfish RL": "#eda100",                # slot 4 yellow
}
# Separate identity axis (device type, not scenario) — deliberately uses slots
# 5-7 so a device-power chart is never confused with a scenario-colored chart.
_DEVICE_COLORS = {
    "ev": "#e87ba4",       # slot 5 magenta
    "battery": "#008300",  # slot 6 green
    "hp": "#4a3aa7",       # slot 7 violet
}
_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
_SURFACE = "#fcfcfb"


def _style_axes(ax) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(axis="y", color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_AXIS)
    ax.tick_params(colors=_SECONDARY_INK, labelsize=9)
    ax.xaxis.label.set_color(_SECONDARY_INK)
    ax.yaxis.label.set_color(_SECONDARY_INK)
    ax.title.set_color(_INK)


def _color_for(scenario: str) -> str:
    return _SCENARIO_COLORS.get(scenario, _MUTED)


def _grouped_bar(summary: list[dict], metric: str, std_metric: str, *, ylabel: str, title: str,
                  out_path: Path, pct: bool = False) -> None:
    """One grouped-bar chart: x = penetration level, one bar per scenario, error bars = std."""
    penetrations = sorted({row["penetration"] for row in summary})
    scenarios = list(dict.fromkeys(row["scenario"] for row in summary))  # first-seen order

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_SURFACE)
    n = len(scenarios)
    width = 0.8 / n
    x = np.arange(len(penetrations))

    for i, scenario in enumerate(scenarios):
        means, stds = [], []
        for pen in penetrations:
            row = next((r for r in summary if r["penetration"] == pen and r["scenario"] == scenario), None)
            # .get(), not [] — summary.json from a run before a metric was added won't have it
            means.append(row.get(metric, 0.0) if row else 0.0)
            stds.append(row.get(std_metric, 0.0) if row else 0.0)
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, means, width * 0.92, yerr=stds, capsize=3, label=scenario,
               color=_color_for(scenario), edgecolor=_SURFACE, linewidth=1.0, zorder=3,
               error_kw=dict(ecolor=_SECONDARY_INK, linewidth=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0%}" for p in penetrations])
    ax.set_xlabel("EV penetration")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    if pct:
        ax.set_ylim(0, 1.05)
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=_SECONDARY_INK, fontsize=9, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _episode_timeline(timelines: list[dict], penetration: float, out_path: Path) -> None:
    """One line per scenario: transformer/feeder loading across the representative day,
    at a single chosen penetration level. Single axis (p.u.) — never dual-axis."""
    rows = [t for t in timelines if t["penetration"] == penetration]
    if not rows:
        return
    steps = range(len(rows[0]["transformer_loading"]))

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_SURFACE)
    for row in rows:
        ax.plot(steps, row["transformer_loading"], label=row["scenario"],
                color=_color_for(row["scenario"]), linewidth=2.0, zorder=3)
    ax.axhline(1.0, color=_MUTED, linestyle="--", linewidth=1.0, zorder=2, label="overload threshold")
    ax.set_xlabel("timestep (15 min)")
    ax.set_ylabel("feeder loading [p.u.]")
    ax.set_title(f"Representative-day feeder loading — {penetration:.0%} EV penetration",
                fontsize=12, fontweight="bold", loc="left")
    _style_axes(ax)
    ax.legend(frameon=False, labelcolor=_SECONDARY_INK, fontsize=9, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _price_and_load(timelines: list[dict], penetration: float, out_path: Path) -> None:
    """Context panel: the shared exogenous day-ahead price + base load every scenario
    at this penetration saw (identical across scenarios — one representative episode
    picked, not overlaid per-scenario, since it's the same input each policy reacted to).
    Two separate axes (own figure each), not one dual-axis chart."""
    rows = [t for t in timelines if t["penetration"] == penetration]
    if not rows:
        return
    row = rows[0]
    steps = range(len(row["price"]))

    fig, (ax_price, ax_load) = plt.subplots(2, 1, figsize=(10, 6), facecolor=_SURFACE, sharex=True)
    ax_price.plot(steps, row["price"], color=_SCENARIO_COLORS["1: flat / immediate"], linewidth=2.0, zorder=3)
    ax_price.set_ylabel("day-ahead price [€/kWh]")
    ax_price.set_title("Episode inputs (shared across all scenarios)", fontsize=12, fontweight="bold", loc="left")
    _style_axes(ax_price)

    ax_load.plot(steps, row["base_load"], color=_SCENARIO_COLORS["2: price-follow (manual)"], linewidth=2.0, zorder=3)
    ax_load.set_ylabel("mean base load [kW]")
    ax_load.set_xlabel("timestep (15 min)")
    _style_axes(ax_load)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _device_behavior(timelines: list[dict], penetration: float, scenario: str, out_path: Path) -> None:
    """What EV/battery/HP actually did for one representative household, across the day,
    for one scenario. The aggregate metrics (comfort rate, kWh cycled) say THAT battery/HP
    did something; this shows WHAT — the trace a per-scenario bar chart can't convey.
    Two panels (power, then SoC) — never one dual-axis chart mixing kW and SoC."""
    row = next((t for t in timelines if t["penetration"] == penetration and t["scenario"] == scenario), None)
    if row is None:
        return
    steps = range(len(row["house_ev_power"]))

    fig, (ax_power, ax_soc) = plt.subplots(2, 1, figsize=(10, 7), facecolor=_SURFACE, sharex=True)

    ax_power.plot(steps, row["house_ev_power"], label="EV", color=_DEVICE_COLORS["ev"], linewidth=2.0, zorder=3)
    ax_power.plot(steps, row["house_battery_power"], label="battery (+charge/−discharge)",
                  color=_DEVICE_COLORS["battery"], linewidth=2.0, zorder=3)
    ax_power.plot(steps, row["house_hp_power"], label="heat pump", color=_DEVICE_COLORS["hp"], linewidth=2.0, zorder=3)
    ax_power.axhline(0.0, color=_AXIS, linewidth=1.0, zorder=2)
    ax_power.set_ylabel("device power [kW]")
    ax_power.set_title(f"One household's device behavior — {scenario}, {penetration:.0%} EV penetration",
                       fontsize=12, fontweight="bold", loc="left")
    _style_axes(ax_power)
    ax_power.legend(frameon=False, labelcolor=_SECONDARY_INK, fontsize=9, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0))

    ax_soc.plot(steps, row["house_ev_soc"], label="EV SoC", color=_DEVICE_COLORS["ev"], linewidth=2.0, zorder=3)
    ax_soc.plot(steps, row["house_battery_soc"], label="battery SoC", color=_DEVICE_COLORS["battery"],
               linewidth=2.0, zorder=3)
    ax_soc.plot(steps, row["house_hp_soc"], label="HP thermal buffer", color=_DEVICE_COLORS["hp"],
               linewidth=2.0, zorder=3)
    ax_soc.set_ylim(-0.05, 1.05)
    ax_soc.set_ylabel("state of charge")
    ax_soc.set_xlabel("timestep (15 min)")
    _style_axes(ax_soc)
    ax_soc.legend(frameon=False, labelcolor=_SECONDARY_INK, fontsize=9, loc="upper left",
                 bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot run_experiment.py results without the dashboard UI")
    parser.add_argument("--results", type=str, default="outputs", help="directory with summary.json/timelines.json")
    parser.add_argument("--out", type=str, default=None, help="directory to write PNGs to (default: <results>/graphs)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out) if args.out else results_dir / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((results_dir / "summary.json").read_text())
    timelines = json.loads((results_dir / "timelines.json").read_text())

    print(f"Plotting {len(summary)} summary rows, {len(timelines)} timelines → {out_dir}")

    _grouped_bar(summary, "curtailment_mean", "curtailment_std",
                ylabel="§14a curtailment events / 24h episode",
                title="Curtailment events by scenario", out_path=out_dir / "curtailment_events.png")
    _grouped_bar(summary, "soc_mean", "soc_std",
                ylabel="fraction of EVs reaching target SoC",
                title="EV SoC satisfaction by scenario", out_path=out_dir / "soc_satisfaction.png", pct=True)
    _grouped_bar(summary, "bill_mean", "bill_std",
                ylabel="mean household bill [€/24h]",
                title="Household electricity bill by scenario", out_path=out_dir / "household_bill.png")
    _grouped_bar(summary, "peak_mean", "peak_std",
                ylabel="peak feeder loading [p.u.]",
                title="Peak feeder loading by scenario", out_path=out_dir / "peak_loading.png")
    # Battery and HP run every scenario too (fixed rule-based, except scenario 3 where
    # they're also learned) — without these the comparison silently looks EV-only.
    _grouped_bar(summary, "hp_comfort_mean", "hp_comfort_std",
                ylabel="fraction of steps thermal buffer ≥ comfort floor",
                title="Heat-pump comfort satisfaction by scenario",
                out_path=out_dir / "hp_comfort_satisfaction.png", pct=True)
    _grouped_bar(summary, "battery_charge_kwh_mean", "battery_charge_kwh_std",
                ylabel="battery energy charged [kWh/24h, feeder total]",
                title="Battery charging by scenario", out_path=out_dir / "battery_charge.png")
    _grouped_bar(summary, "battery_discharge_kwh_mean", "battery_discharge_kwh_std",
                ylabel="battery energy discharged [kWh/24h, feeder total]",
                title="Battery discharging by scenario", out_path=out_dir / "battery_discharge.png")

    penetrations = sorted({row["penetration"] for row in summary})
    highest = penetrations[-1]
    _episode_timeline(timelines, highest, out_dir / f"episode_timeline_{int(highest*100)}pct.png")
    _price_and_load(timelines, highest, out_dir / f"episode_inputs_{int(highest*100)}pct.png")
    # RL (scenario 3) is the only scenario where battery/HP are actually learned, not a
    # fixed rule — that's the one where "what did they do" is the interesting question.
    rl_scenarios = {row["scenario"] for row in summary if row["scenario"].startswith("3:")}
    for scenario in rl_scenarios:
        _device_behavior(timelines, highest, scenario,
                         out_dir / f"device_behavior_{int(highest*100)}pct.png")

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} PNGs in {out_dir}")


if __name__ == "__main__":
    main()
