# dashboard/app.py
# ─────────────────────────────────────────────────────────────
# Streamlit dashboard for the GridKIT §14a curtailment comparison.
#
# It reads results from two producers, which write the same schema:
#   · runs/<id>/          one run started from the map (scripts/train_run.py)
#   · outputs/            the scenario × penetration batch sweep
#                           (scripts/run_experiment.py)
#
# Entry points:
#   streamlit run src/GridKIT/scripts/app.py    map + dashboard in one app
#   streamlit run src/GridKIT/dashboard/app.py  the dashboard on its own
#
# `render_dashboard()` is the page scripts/app.py composes: it browses saved
# runs and renders the selected one. It never starts or writes a run (except
# deleting one on request), so it can be refreshed freely while train_run.py
# is writing into the same directory.
#
# This module is only the shell: it loads the result files, renders the
# glossary and hands each tab its data. Every view lives in its own module
# (overview, utilization, devices, comparison, training) and every colour and
# label in theme.py.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── path bootstrap (so `streamlit run` finds the packages) ──
# Needed because this file is run directly as a script, not imported as part of
# the package: without it `from dashboard...` below fails unless PYTHONPATH
# happens to be set outside. Same bootstrap as scripts/grid_designer.py.
_PKG_DIR = Path(__file__).resolve().parent.parent   # src/GridKIT/
_SRC_DIR = _PKG_DIR.parent                          # src/
for _p in (str(_PKG_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from core import run_store as rs
from dashboard import theme
from dashboard.utilization import render_utilization
from dashboard.devices import render_devices
from dashboard.training import render_training
from dashboard.overview import render_overview
from dashboard.comparison import render_comparison

OUTPUT_DIR = Path(os.environ.get("GRIDKIT_OUTPUT_DIR", "outputs"))

#: Session-state key for the selected run, so the pick survives a rerun.
RUN_STATE = "dashboard_selected_run"

#: run_store's states are persisted values compared against status.json on
#: disk, so they stay English; only their on-screen labels are translated.
STATE_LABELS_DE = {
    rs.QUEUED: "wartet",
    rs.RUNNING: "läuft",
    rs.DONE: "fertig",
    rs.FAILED: "fehlgeschlagen",
}

def _load(name: str):
    """Read one result file; None when it is missing, unreadable or malformed.

    A run killed mid-write leaves truncated JSON behind, and that used to take
    the whole dashboard down with a raw parser traceback. Returning None lets
    each view say what is missing instead.
    """
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.error(
            f"`{name}` konnte nicht gelesen werden ({type(exc).__name__}). "
            "Die Datei ist vermutlich unvollständig; das passiert, wenn ein Lauf "
            "abgebrochen wurde. Experiment erneut ausführen."
        )
        return None


def render_results(summary, timelines, network=None, output_dir=None) -> None:
    """Render the scenario comparison dashboard from data already in memory.

    `network` ist optional: wenn übergeben, wird zusätzlich die Überlastungskarte
    gezeichnet. `output_dir` sagt, wo die Trainingsmetriken dieses Laufs liegen:
    für einen Lauf aus der Karte sein eigenes Verzeichnis, sonst OUTPUT_DIR.
    Ohne diesen Parameter zeigte der Training-Reiter eines Karten-Laufs die
    Metriken des Batch-Experiments, also die eines ganz anderen Netzes.
    """
    if not summary:
        st.info("Für diese Auswahl liegen noch keine Ergebnisse vor.")
        return

    _render_glossary()

    tab_overview, tab_utilization, tab_devices, tab_comparison, tab_training = st.tabs(
        ["Überblick", "Netzauslastung", "Geräte & Haushalte", "Szenarienvergleich", "Training"]
    )
    with tab_overview:
        render_overview(summary, timelines)
    with tab_utilization:
        render_utilization(timelines or [])
    with tab_devices:
        render_devices(summary, timelines or [])
    with tab_comparison:
        render_comparison(summary, timelines, network)
    with tab_training:
        render_training(output_dir if output_dir is not None else OUTPUT_DIR, summary)


def _render_glossary() -> None:
    """The terms without which no number on this page can be read."""
    with st.expander("Wie lese ich das? Begriffe in einem Satz"):
        left, right = st.columns(2)
        left.markdown(
            "**Auslastung (%)**: Belastung im Verhältnis zur Nennleistung. "
            "100 % heißt genau ausgelastet, darüber ist Überlast.\n\n"
            "**Transformator vs. Leitung**: beide können überlasten. Im "
            "Niederspannungsnetz erreicht meist das **Kabel** zuerst seine "
            "Grenze, während der Transformator noch entspannt aussieht.\n\n"
            "**Ausstattungsgrad**: Anteil der Haushalte mit flexiblen Geräten. Im "
            "Batch-Experiment bekommen genau diese Haushalte die volle Ausstattung "
            "(E-Auto, Batterie, Wärmepumpe, PV), die übrigen keines davon. "
            "Der Härtegrad des Tests."
        )
        right.markdown(
            "**§14a EnWG**: erlaubt dem Netzbetreiber, steuerbare Geräte "
            "gedrosselt zu betreiben, wenn das Netz sonst überlastet. Ein "
            "„Eingriff“ ist eine Viertelstunde, in der das passiert.\n\n"
            "**Ladestand (SoC)**: Füllstand von Autobatterie oder Speicher, "
            "0 bis 1.\n\n"
            "**Szenarien**: die vier Regelstrategien, die hier verglichen werden. "
            "Was jede einzelne tut, steht unten."
        )
        _render_scenario_glossary()


def _render_scenario_glossary() -> None:
    """The four strategies, one sentence each. Full width below the two
    columns, because four entries would make one column twice as long."""
    st.markdown("**Die vier Szenarien im Einzelnen**")
    st.markdown("\n".join(
        f"- **{theme.scenario_short(s)}**: {theme.scenario_description(s)}"
        for s in theme.SCENARIO_ORDER
    ))
    st.caption(
        "Bei den ersten drei Strategien sind Batterie (Eigenverbrauch) und Wärmepumpe "
        "(Thermostat) fest geregelt, unterschieden wird nur das Laden des E-Autos. "
        "Beim RL lernen alle drei Geräte mit."
    )


SUBTITLE = (
    "Jeder Haushalt betreibt drei steuerbare Geräte-Agenten (EV, Batterie, Wärmepumpe) "
    "sowie eine exogene Dach-PV-Anlage, auf Basis realer, wetterabhängiger Profile "
    "(GridCreator/pyCity). Der Mechanismus wird unter vereinfachten Annahmen gezeigt "
    "(Ersatz-Lastfluss). Ein relativer Vergleich, keine Prognose."
)


def _run_label(record: dict) -> str:
    """One line per run for the picker: name, state and whether it has results."""
    state = STATE_LABELS_DE.get(record["state"], record["state"])
    mark = " ✓ Ergebnisse" if record.get("has_results") else ""
    return f"{record.get('name', '')} · {state}{mark}  ·  {record['run_id']}"


def render_dashboard() -> None:
    """The dashboard page: browse saved runs and render the selected one.

    No `st.set_page_config` here, so scripts/app.py can mount this page next to
    the map; `main()` below owns the page config for standalone use.
    """
    st.title("GridKIT Dashboard")
    st.caption(SUBTITLE)

    if st.button("🔄 Aktualisieren", help="Läufe neu einlesen; ein laufendes Training "
                                          "schreibt weiter, während diese Seite offen ist"):
        st.rerun()

    runs = rs.list_runs()
    if not runs:
        st.info(
            "Noch keine Läufe. Auf der Seite **Karte** ein Gebiet auswählen, die Haushalte "
            "konfigurieren und „Speichern & Training starten“. Der Lauf erscheint dann hier."
        )
        return

    ids = [r["run_id"] for r in runs]
    by_id = {r["run_id"]: r for r in runs}
    # Remember the pick across reruns, but never point at a deleted run.
    if st.session_state.get(RUN_STATE) not in ids:
        st.session_state[RUN_STATE] = ids[0]
    sel = st.selectbox("Lauf", ids, key=RUN_STATE, format_func=lambda i: _run_label(by_id[i]))
    record = by_id[sel]

    head, delete = st.columns([5, 1])
    head.caption(f"**{record.get('name', '')}** · {record.get('n_households', '?')} Haushalte · "
                 f"erstellt {record.get('created', '?')} · `{sel}`")
    if delete.button("🗑 Löschen"):
        rs.delete_run(sel)
        st.session_state.pop(RUN_STATE, None)
        st.rerun()

    state = record["state"]
    if state == rs.QUEUED:
        st.info("Gespeichert, das Training hat noch nicht begonnen.")
        return
    if state == rs.RUNNING and record.get("stale"):
        st.warning(
            f"Seit über {rs.STALE_AFTER_SECONDS // 60} Minuten kein Fortschritt "
            f"(zuletzt: „{record.get('message', '')}“). Der Trainingsprozess ist "
            "wahrscheinlich abgestürzt, etwa aus Speichermangel, wenn mehrere Läufe "
            "gleichzeitig trainiert haben. Am besten löschen und neu starten."
        )
    elif state == rs.RUNNING:
        st.progress(min(1.0, record.get("progress", 0.0)), text=record.get("message", "Training läuft…"))
        st.caption("Das Training läuft noch. Auf „Aktualisieren“ klicken für den neuesten Stand; "
                   "die Ergebnisse erscheinen hier, sobald es fertig ist.")
    elif state == rs.FAILED:
        st.error(f"Fehlgeschlagen: {record.get('message', '')}")
        st.caption(f"Einzelheiten in `runs/{sel}/train.log`.")

    if not record.get("has_results"):
        return

    summary, timelines = rs.load_results(sel)
    try:
        network = rs.load_network(sel)
    except Exception:
        # A run saved by an older version, or a half-written file: the overload
        # map is the only view that needs the topology, so lose just that one.
        network = None
    render_results(summary, timelines, network=network, output_dir=rs.run_dir(sel))


def render_batch() -> None:
    """The batch experiment's output, read from OUTPUT_DIR."""
    st.title("GridKIT Dashboard")
    st.caption(SUBTITLE)
    summary = _load("summary.json")
    timelines = _load("timelines.json")
    if not summary:
        st.warning(f"Keine Ergebnisse in `{OUTPUT_DIR}/`. Zuerst "
                   "`python -m GridKIT.scripts.run_experiment` ausführen.")
        return
    render_results(summary, timelines, output_dir=OUTPUT_DIR)


def main() -> None:
    st.set_page_config(page_title="GridKIT Dashboard", layout="wide")
    # Two producers can have written results. Ask only when both actually have
    # something to show, so the usual case stays a single click-free page.
    has_batch = (OUTPUT_DIR / "summary.json").exists()
    has_runs = bool(rs.list_runs())
    if has_batch and has_runs:
        source = st.sidebar.radio(
            "Datenquelle", ["Lauf aus der Karte", f"Batch-Experiment ({OUTPUT_DIR})"],
            help="Läufe aus der Karte liegen unter runs/, das Batch-Experiment unter outputs/.",
        )
        # A statement, not a conditional expression: Streamlit's "magic" renders
        # any bare expression in the main script, and both calls return None, so
        # the expression form printed a stray "None" onto the page.
        if source == "Lauf aus der Karte":
            render_dashboard()
        else:
            render_batch()
    elif has_batch:
        render_batch()
    else:
        render_dashboard()


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    main()
elif _running_under_streamlit():
    # `streamlit run` imports this module (name != __main__), so run automatically
    # only in that case, NOT on a plain import (e.g. when another app imports
    # render_results).
    main()
