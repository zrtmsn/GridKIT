# dashboard/app.py
# ─────────────────────────────────────────────────────────────
# Streamlit dashboard for the GridKIT scenario × penetration experiment.
# Reads outputs/summary.json + outputs/timelines.json (produced by
# scripts/run_experiment.py) and renders the §14a curtailment comparison.
#
# Run:  streamlit run src/GridKIT/dashboard/app.py
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

OUTPUT_DIR = Path(os.environ.get("GRIDKIT_OUTPUT_DIR", "outputs"))
EPISODE_START_HOUR = 12
TIMESTEP_HOURS = 0.25

SCENARIO_ORDER = [
    "1: flat / immediate",
    "2: price-follow (manual)",
    "2: price-follow (automated)",
    "3: selfish RL",
]
SCENARIO_COLORS = {
    "1: flat / immediate": "#6c757d",
    "2: price-follow (manual)": "#f4a261",
    "2: price-follow (automated)": "#e76f51",
    "3: selfish RL": "#2a9d8f",
}


def _load(name: str):
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _hours_axis(n: int) -> np.ndarray:
    return (EPISODE_START_HOUR + np.arange(n) * TIMESTEP_HOURS)


def main() -> None:
    st.set_page_config(page_title="GridKIT — §14a under price-reactive charging", layout="wide")
    st.title("GridKIT — does §14a curtailment hold up as charging gets price-reactive?")
    st.caption(
        "Demonstrated mechanism under simplified inputs (stub feeder, surrogate power flow) — "
        "relative comparison, not a real-grid forecast."
    )

    summary = _load("summary.json")
    timelines = _load("timelines.json")
    if summary is None:
        st.warning(f"No results in `{OUTPUT_DIR}/`. Run `python -m GridKIT.scripts.run_experiment` first.")
        st.stop()

    df = pd.DataFrame(summary)
    penetrations = sorted(df["penetration"].unique())

    # ── 1. Curtailment comparison across penetration ──────────
    st.header("Curtailment events by scenario and EV penetration")
    st.write(
        "Timesteps per 24 h episode where §14a dimming was triggered (mean ± std over seeds). "
        "The story: naive **automated** price-following synchronizes into the cheap overnight "
        "window and curtails most; the selfish **RL** agent, able to sense local voltage / past "
        "dimming, learns to spread out."
    )
    fig, ax = plt.subplots(figsize=(9, 4.2))
    scenarios = [s for s in SCENARIO_ORDER if s in df["scenario"].unique()]
    x = np.arange(len(penetrations))
    width = 0.8 / max(1, len(scenarios))
    for i, scen in enumerate(scenarios):
        sub = df[df["scenario"] == scen].set_index("penetration").reindex(penetrations)
        ax.bar(x + i * width, sub["curtailment_mean"], width,
               yerr=sub["curtailment_std"], capsize=3,
               label=scen, color=SCENARIO_COLORS.get(scen, None))
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels([f"{p:.0%}" for p in penetrations])
    ax.set_xlabel("EV penetration")
    ax.set_ylabel("Curtailment events / episode")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

    # ── 2. SoC satisfaction (the customer-side cost) ──────────
    st.header("SoC satisfaction — did customers get charged in time?")
    pivot = df.pivot_table(index="penetration", columns="scenario", values="soc_mean")
    pivot = pivot.reindex(columns=[s for s in SCENARIO_ORDER if s in pivot.columns])
    pivot.index = [f"{p:.0%}" for p in pivot.index]
    st.dataframe(pivot.style.format("{:.2f}").background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                 width="stretch")

    # ── 3. Representative 24 h timeline ───────────────────────
    if timelines:
        st.header("A representative 24 h episode")
        tdf = pd.DataFrame(timelines)
        c1, c2 = st.columns(2)
        sel_pen = c1.selectbox("EV penetration", penetrations, format_func=lambda p: f"{p:.0%}")
        avail = [s for s in SCENARIO_ORDER if s in tdf[tdf["penetration"] == sel_pen]["scenario"].values]
        sel_scen = c2.selectbox("Scenario", avail)

        row = tdf[(tdf["penetration"] == sel_pen) & (tdf["scenario"] == sel_scen)]
        if not row.empty:
            rec = row.iloc[0]
            hours = _hours_axis(len(rec["transformer_loading"]))

            fig2, (axa, axb) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axa.plot(hours, rec["transformer_loading"], label="transformer loading", color="#264653")
            axa.plot(hours, rec["max_line_loading"], label="max line loading", color="#e76f51")
            axa.axhline(1.0, ls="--", color="red", lw=1, label="overload threshold")
            curt = np.array(rec["curtailment"], dtype=bool)
            axa.fill_between(hours, 0, 1.4, where=curt, color="red", alpha=0.12, label="§14a curtailing")
            axa.set_ylabel("loading (p.u.)")
            axa.set_ylim(0, 1.4)
            axa.legend(fontsize=8, ncol=2)
            axa.grid(alpha=0.3)
            axa.set_title(f"{sel_scen} @ {sel_pen:.0%} penetration")

            axb.plot(hours, rec["price"], label="price (€/kWh)", color="#2a9d8f")
            axb.set_ylabel("price (€/kWh)", color="#2a9d8f")
            axb.tick_params(axis="y", labelcolor="#2a9d8f")
            axc = axb.twinx()
            axc.plot(hours, rec["base_load"], label="base load (kW)", color="#6c757d", ls=":")
            axc.set_ylabel("base load (kW)", color="#6c757d")
            axb.set_xlabel("hour of day")
            axb.grid(alpha=0.3)
            st.pyplot(fig2)
            st.metric("SoC satisfaction (this episode)", f"{rec['soc_satisfaction_rate']:.0%}")

    with st.expander("Raw summary table"):
        st.dataframe(df, width="stretch")


if __name__ == "__main__":
    main()
else:
    # `streamlit run` executes the module top-level
    main()
