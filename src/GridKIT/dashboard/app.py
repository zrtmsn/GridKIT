# dashboard/app.py
# ─────────────────────────────────────────────────────────────
# GridKIT — Dashboard.
#
# Reads runs saved via core.run_store (created by map_ui/map_widget.py's
# "Speichern & Training starten") and renders their status / results.
# Pure viewer — never launches or writes to a run itself (except deleting one
# on request), so it can be reloaded/refreshed freely while scripts.train_run
# is writing to the same run in the background.
#
# Run:  PYTHONPATH=src/GridKIT:src streamlit run src/GridKIT/dashboard/app.py
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path

# ── path bootstrap (so `streamlit run` finds the packages) ──
_SCRIPT_DIR = Path(__file__).resolve().parent.parent   # src/GridKIT/
_SRC_DIR = _SCRIPT_DIR.parent                            # src/
for _p in (str(_SCRIPT_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from core import run_store as rs

SCENARIO_ORDER = [
    "1: flat / immediate",
    "2: price-follow (automated)",
    "3: selfish RL",
]
SCENARIO_COLORS = {
    "1: flat / immediate": "#6c757d",
    "2: price-follow (automated)": "#e76f51",
    "3: selfish RL": "#2a9d8f",
}


def render_dashboard() -> None:
    """Page body — no st.set_page_config here, so this can be composed as one
    page of a larger app (see scripts/app.py) as well as run standalone
    (see main() below, which owns page config for the standalone case)."""
    st.title("GridKIT — Dashboard")
    st.caption(
        "Zeigt gespeicherte Netze und Trainingsläufe aus der Karte "
        "(erreichbar über `streamlit run src/GridKIT/scripts/app.py`, oder "
        "eigenständig über `streamlit run src/GridKIT/map_ui/map_widget.py`)."
    )

    if st.button("🔄 Aktualisieren"):
        st.rerun()

    runs = rs.list_runs()
    if not runs:
        st.info("Noch keine gespeicherten Läufe. Im Karten-UI ein Netz speichern und trainieren.")
        return

    st.subheader("Gespeicherte Läufe")
    st.dataframe(
        [{"Lauf": r["run_id"], "Name": r.get("name", ""), "Status": r["state"],
          "Fortschritt": f"{r.get('progress', 0) * 100:.0f}%", "Haushalte": r.get("n_households", 0),
          "Erstellt": r.get("created", ""), "Ergebnisse": "✓" if r.get("has_results") else "—"}
         for r in runs],
        width="stretch", hide_index=True,
    )

    ids = [r["run_id"] for r in runs]
    by_id = {r["run_id"]: r for r in runs}
    sel = st.selectbox(
        "Lauf auswählen", ids,
        format_func=lambda i: f"{by_id[i].get('name', '')} · {by_id[i]['state']} ({i})",
    )
    r = by_id[sel]

    top_col, delete_col = st.columns([4, 1])
    top_col.caption(f"**{sel}** — {r.get('n_households', '?')} Haushalte · erstellt {r.get('created', '?')}")
    if delete_col.button("🗑 Lauf löschen"):
        rs.delete_run(sel)
        st.rerun()

    state = r["state"]
    if state == rs.QUEUED:
        st.info("Gespeichert, Training wurde noch nicht gestartet.")
    elif state == rs.RUNNING and r.get("stale"):
        st.warning(
            f"Zeigt seit über {rs.STALE_AFTER_SECONDS // 60} Minuten keinen Fortschritt mehr "
            f"(letzte Meldung: „{r.get('message', '')}“) — der Prozess ist wahrscheinlich abgestürzt "
            "(z. B. durch zu wenig Arbeitsspeicher, wenn mehrere Trainings gleichzeitig liefen), nicht "
            "wirklich noch am Trainieren. Sicherheitshalber löschen und neu starten."
        )
    elif state == rs.RUNNING:
        st.progress(min(1.0, r.get("progress", 0.0)), text=r.get("message", "Training läuft…"))
        st.caption("Noch am Trainieren — auf „Aktualisieren“ klicken, um den Fortschritt zu sehen.")
    elif state == rs.FAILED:
        st.error(f"Fehlgeschlagen: {r.get('message', '')}")
    elif r.get("has_results"):
        summary, timelines = rs.load_results(sel)
        render_results(summary, timelines)
    else:
        st.info(f"Status: {state}")


def render_results(summary: list[dict], timelines: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    st.subheader("Szenarienvergleich")
    present = {row["scenario"] for row in summary}
    scenarios = [s for s in SCENARIO_ORDER if s in present]
    by_scenario = {row["scenario"]: row for row in summary}
    x = np.arange(len(scenarios))
    colors = [SCENARIO_COLORS.get(s, "#888888") for s in scenarios]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(x, [by_scenario[s]["curtailment_mean"] for s in scenarios],
                yerr=[by_scenario[s]["curtailment_std"] for s in scenarios], capsize=3, color=colors)
    axes[0].set_title("§14a-Abregelungen / Episode")

    axes[1].bar(x, [by_scenario[s]["soc_mean"] for s in scenarios],
                yerr=[by_scenario[s]["soc_std"] for s in scenarios], capsize=3, color=colors)
    axes[1].set_title("EV-Ladeziel erreicht")
    axes[1].set_ylim(0, 1.05)

    axes[2].bar(x, [by_scenario[s]["reward_mean"] for s in scenarios],
                yerr=[by_scenario[s]["reward_std"] for s in scenarios], capsize=3, color=colors)
    axes[2].set_title("Reward (Ø)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Rohdaten (summary.json)"):
        st.dataframe(summary, width="stretch")

    st.subheader("Repräsentative 24h-Episode")
    available = [s for s in SCENARIO_ORDER if s in (timelines or {})]
    if not available:
        st.caption("Keine Zeitreihen für diesen Lauf gespeichert.")
        return
    chosen = st.selectbox("Szenario", available)
    tl = timelines[chosen]

    n = len(tl["transformer_loading"])
    hours = 12 + np.arange(n) * 0.25  # episode starts at noon

    fig2, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(hours, tl["transformer_loading"], color="#264653", label="Trafo-Auslastung (p.u.)")
    ax1.axhline(1.0, color="red", ls="--", lw=1, label="Überlastschwelle")
    curt = np.array(tl["curtailment"], dtype=bool)
    ax1.fill_between(hours, 0, 1.4, where=curt, color="red", alpha=0.12, label="Abregelung aktiv")
    ax1.set_ylim(0, 1.4)
    ax1.set_ylabel("Auslastung (p.u.)")

    ax2 = ax1.twinx()
    ax2.plot(hours, tl["price"], color="#2a9d8f", alpha=0.6, label="Preis (€/kWh)")
    ax2.set_ylabel("Preis (€/kWh)", color="#2a9d8f")

    ax1.set_xlabel("Uhrzeit (Episode läuft Mittag → Mittag)")
    ax1.set_title(f"{chosen} — ein repräsentativer Tag")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)


def main() -> None:
    """Standalone entry point: streamlit run src/GridKIT/dashboard/app.py"""
    st.set_page_config(page_title="GridKIT — Dashboard", layout="wide")
    render_dashboard()


if __name__ == "__main__":
    main()
