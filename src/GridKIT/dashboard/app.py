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


# ══════════════════════════════════════════════════════════════
# Overload map — WHERE the grid hurt, not just how often
# ══════════════════════════════════════════════════════════════
#: Loading (p.u.) → colour. A real LV grid is cable-limited long before the
#: transformer notices, so the map has to make a 0.9 pu cable visible.
_LOAD_COLORS = ((1.0, "#d7191c", 5.0), (0.9, "#fdae61", 4.0), (0.7, "#ffd54f", 3.0), (0.0, "#7cb342", 2.5))


def _load_style(pu: float, tripped: bool = False) -> tuple[str, float]:
    """(colour, line weight) for a loading in p.u.

    `tripped` forces the overload colour. The stored peak is a MEAN over seeds, so a
    cable that violated its rating in two runs out of six averages below 1.0 — it must
    still read as a violation, or the map would contradict the overload count beside it.
    """
    if tripped:
        return _LOAD_COLORS[0][1], _LOAD_COLORS[0][2]
    for floor, color, weight in _LOAD_COLORS:
        if pu > floor:
            return color, weight
    return "#7cb342", 2.5


def render_overload_map(network, summary, key: str = "overload_map") -> None:
    """Draw the grid, highlighting the cables that ran hot or tripped.

    `network` is the run's stored GridNetwork; `summary` supplies the per-line peak
    loading recorded for each scenario. Unlisted lines stayed below the watch level
    and are drawn as faint background, so the eye goes straight to the hot path.
    """
    import folium
    from streamlit_folium import st_folium

    if network is None or not getattr(network, "buses", None):
        st.info("No stored network for this run — the overload map needs `network.json`.")
        return

    rows = [r for r in summary if r.get("line_peak_loading_pu")]
    if not rows:
        st.info("This run predates per-line recording, so there is no overload map for it. "
                "New runs store it automatically.")
        return

    labels = [r["scenario"] for r in rows]
    chosen = st.selectbox("Scenario", labels, index=len(labels) - 1, key=f"{key}_scenario")
    row = next(r for r in rows if r["scenario"] == chosen)
    peaks: dict = row.get("line_peak_loading_pu") or {}
    steps: dict = row.get("line_overload_steps") or {}

    coords = {b.bus_id: (b.y_coord, b.x_coord)
              for b in network.buses if b.x_coord is not None and b.y_coord is not None}
    if not coords:
        st.info("The stored network has no coordinates, so it cannot be mapped.")
        return

    centre = (float(np.mean([c[0] for c in coords.values()])),
              float(np.mean([c[1] for c in coords.values()])))
    m = folium.Map(location=centre, zoom_start=15, tiles="cartodbpositron")

    n_over = 0
    for ln in network.lines:
        a, b = coords.get(ln.from_bus), coords.get(ln.to_bus)
        if not a or not b:
            continue
        pu = peaks.get(ln.line_id)
        if pu is None:
            folium.PolyLine([a, b], color="#c3c9d1", weight=1.2, opacity=0.5).add_to(m)
            continue
        trips = steps.get(ln.line_id, 0)
        color, weight = _load_style(pu, tripped=trips > 0)
        n_over += trips > 0
        folium.PolyLine(
            [a, b], color=color, weight=weight, opacity=0.95,
            tooltip=f"{ln.line_id} — peak {pu:.2f} pu" + (f", overloaded {trips:.1f} steps" if trips else ""),
        ).add_to(m)

    for t in network.transformers:
        tb = coords.get(t.lv_bus)
        if tb:
            folium.Marker(tb, icon=folium.Icon(color="black", icon="bolt", prefix="fa"),
                          tooltip=f"{t.trafo_id} — {t.s_nom_mva*1000:.0f} kVA").add_to(m)

    worst = max(peaks.values()) if peaks else 0.0
    st.caption(
        f"**{chosen}** — worst cable {worst:.2f} pu · {n_over} cable(s) exceeded rating. "
        "🟥 >1.0 overloaded · 🟧 >0.9 · 🟨 >0.7 · 🟩 loaded but healthy · grey = below watch level. "
        "⚡ = transformer. Hover any segment for its peak."
    )
    st_folium(m, height=520, width=None, returned_objects=[], key=key)


def render_results(summary, timelines, network=None) -> None:
    """Render the scenario-comparison dashboard from in-memory data.

    Reused by the unified web app (per-run results), as well as the standalone
    dashboard `main()` which loads the batch experiment's outputs/. `network` is
    optional: when supplied, the overload map is drawn too.
    """
    if not summary:
        st.info("No results for this selection yet.")
        return

    df = pd.DataFrame(summary)
    penetrations = sorted(df["penetration"].unique())

    if network is not None:
        st.header("Where the grid was overloaded")
        render_overload_map(network, summary)

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

            has_devices = "ev_power" in rec and rec["ev_power"] is not None
            n_panels = 4 if has_devices else 3
            fig2, axes = plt.subplots(n_panels, 1, figsize=(10, 2.6 * n_panels), sharex=True)
            axa, axb, axd = axes[0], axes[1], axes[2]
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

            # price + PV generation (why self-consume vs export matters)
            axb.plot(hours, rec["price"], label="price (€/kWh)", color="#2a9d8f")
            axb.set_ylabel("price (€/kWh)", color="#2a9d8f")
            axb.tick_params(axis="y", labelcolor="#2a9d8f")
            if "pv_generation" in rec and rec["pv_generation"] is not None:
                axpv = axb.twinx()
                axpv.fill_between(hours, 0, rec["pv_generation"], color="#e9c46a", alpha=0.4, label="PV (kW)")
                axpv.set_ylabel("PV generation (kW)", color="#b8860b")
                axpv.tick_params(axis="y", labelcolor="#b8860b")
            axb.grid(alpha=0.3)

            # base load + outdoor temperature (heat-pump driver)
            axd.plot(hours, rec["base_load"], label="base load (kW)", color="#6c757d", ls=":")
            axd.set_ylabel("base load (kW)", color="#6c757d")
            if "temperature" in rec and rec["temperature"] is not None:
                axt = axd.twinx()
                axt.plot(hours, rec["temperature"], label="temperature (°C)", color="#e76f51")
                axt.set_ylabel("temperature (°C)", color="#e76f51")
                axt.tick_params(axis="y", labelcolor="#e76f51")
            axd.grid(alpha=0.3)

            # per-device decisions: what each device type chose (feeder-aggregate delivered kW)
            if has_devices:
                axe = axes[3]
                axe.plot(hours, rec["ev_power"], label="EV charging", color="#457b9d")
                axe.plot(hours, rec["hp_power"], label="heat pump", color="#e63946")
                axe.plot(hours, rec["battery_power"], label="battery (+chg / −dis)", color="#2a9d8f")
                if "pv_generation" in rec and rec["pv_generation"] is not None:
                    axe.plot(hours, [-p for p in rec["pv_generation"]], label="PV (−gen)",
                             color="#e9c46a", ls="--", alpha=0.7)
                axe.axhline(0, color="k", lw=0.6)
                axe.set_ylabel("device power (kW)")
                axe.legend(fontsize=8, ncol=2)
                axe.grid(alpha=0.3)
                axe.set_title("Per-device decisions (feeder total)", fontsize=9)
            # Episode runs noon→noon, so the raw axis is 12..36. Relabel to clock
            # time (mod 24) and mark midnight so it reads intuitively.
            import matplotlib.ticker as mticker
            for ax in axes:
                ax.axvline(24, color="k", ls=":", lw=0.9, alpha=0.5)
            axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(6))
            axes[-1].xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda h, _: f"{int(round(h)) % 24:02d}:00"))
            axes[-1].set_xlabel("time of day  (episode runs noon → midnight ┊ → noon next day)")
            st.pyplot(fig2)

            # ── one representative household: exact device power + EV availability + SoC ──
            if "house_ev_power" in rec and rec["house_ev_power"] is not None:
                st.subheader("A single representative household — exact device power")
                st.caption(
                    "One actual home's device decisions (not the feeder sum above). The shaded band "
                    "marks when its EV is home / plugged in; the lower panel tracks each device's state."
                )
                figh, (hp1, hp2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                hp1.plot(hours, rec["house_ev_power"], label="EV charging", color="#457b9d")
                hp1.plot(hours, rec["house_hp_power"], label="heat pump", color="#e63946")
                hp1.plot(hours, rec["house_battery_power"], label="battery (+chg / −dis)", color="#2a9d8f")
                hp1.plot(hours, [-p for p in rec["house_pv"]], label="PV (−gen)", color="#e9c46a", ls="--", alpha=0.7)
                hp1.axhline(0, color="k", lw=0.6)
                ymin, ymax = hp1.get_ylim()
                avail = np.array(rec["house_ev_available"], dtype=bool)
                hp1.fill_between(hours, ymin, ymax, where=avail, color="#457b9d", alpha=0.08,
                                 label="EV at home / plugged in")
                hp1.set_ylim(ymin, ymax)
                hp1.set_ylabel("device power (kW)")
                hp1.legend(fontsize=8, ncol=2)
                hp1.grid(alpha=0.3)

                hp2.plot(hours, rec["house_ev_soc"], label="EV SoC", color="#457b9d")
                hp2.axhline(0.80, ls=":", color="#457b9d", lw=1, alpha=0.7)   # EV target
                hp2.plot(hours, rec["house_battery_soc"], label="battery SoC", color="#2a9d8f")
                hp2.plot(hours, rec["house_hp_soc"], label="HP thermal buffer", color="#e63946")
                hp2.axhline(0.30, ls=":", color="#e63946", lw=1, alpha=0.7)   # HP comfort floor
                hp2.set_ylabel("state of charge")
                hp2.set_ylim(0, 1.05)
                hp2.legend(fontsize=8, ncol=3)
                hp2.grid(alpha=0.3)

                for ax in (hp1, hp2):
                    ax.axvline(24, color="k", ls=":", lw=0.9, alpha=0.5)
                hp2.xaxis.set_major_locator(mticker.MultipleLocator(6))
                hp2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda h, _: f"{int(round(h)) % 24:02d}:00"))
                hp2.set_xlabel("time of day  (noon → noon next day)")
                st.pyplot(figh)
            st.metric("SoC satisfaction (this episode)", f"{rec['soc_satisfaction_rate']:.0%}")

    with st.expander("Raw summary table"):
        st.dataframe(df, width="stretch")


def main() -> None:
    st.set_page_config(page_title="GridKIT — §14a under multi-device flexibility", layout="wide")
    st.title("GridKIT — does §14a curtailment hold up with EV + battery + heat pump + PV?")
    st.caption(
        "Each household runs three controllable device-agents (EV, battery, heat pump) plus "
        "exogenous rooftop PV, on real weather-driven profiles (GridCreator/pyCity). Demonstrated "
        "mechanism under simplified inputs (surrogate power flow) — relative comparison, not a forecast."
    )
    summary = _load("summary.json")
    timelines = _load("timelines.json")
    if not summary:
        st.warning(f"No results in `{OUTPUT_DIR}/`. Run `python -m GridKIT.scripts.run_experiment` first.")
        return
    render_results(summary, timelines)


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    main()
elif _running_under_streamlit():
    # `streamlit run` imports the module (name != __main__); auto-run only then,
    # NOT on a plain import (e.g. the unified web app importing render_results).
    main()
