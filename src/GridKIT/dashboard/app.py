# dashboard/app.py
# ─────────────────────────────────────────────────────────────
# Streamlit dashboard for the GridKIT scenario × penetration experiment.
# Reads outputs/summary.json + outputs/timelines.json (produced by
# scripts/run_experiment.py) and renders the §14a curtailment comparison.
#
# Run:  streamlit run src/GridKIT/dashboard/app.py
#
# This module is now only the shell: it loads the result files, renders the
# glossary and hands each tab its data. Every view lives in its own module
# (ueberblick, auslastung, geraete, vergleich, training) and every colour and
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

from dashboard.auslastung import render_auslastung
from dashboard.geraete import render_geraete
from dashboard.training import render_training
from dashboard.ueberblick import render_ueberblick
from dashboard.vergleich import render_vergleich

OUTPUT_DIR = Path(os.environ.get("GRIDKIT_OUTPUT_DIR", "outputs"))
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


def render_results(summary, timelines, network=None) -> None:
    """Rendert das Szenario-Vergleichs-Dashboard aus Daten im Arbeitsspeicher.

    Wird sowohl von der einheitlichen Web-App (Ergebnisse je Lauf) als auch vom
    eigenständigen Dashboard `main()` genutzt, das die Ausgabe des Batch-Experiments
    lädt. `network` ist optional: wenn übergeben, wird zusätzlich die Überlastungskarte
    gezeichnet.
    """
    if not summary:
        st.info("Für diese Auswahl liegen noch keine Ergebnisse vor.")
        return

    _render_glossary()

    tab_ueberblick, tab_last, tab_geraete, tab_vergleich, tab_training = st.tabs(
        ["Überblick", "Netzauslastung", "Geräte & Haushalte", "Szenarienvergleich", "Training"]
    )
    with tab_ueberblick:
        render_ueberblick(summary, timelines)
    with tab_last:
        render_auslastung(timelines or [])
    with tab_geraete:
        render_geraete(summary, timelines or [])
    with tab_vergleich:
        render_vergleich(summary, timelines, network)
    with tab_training:
        render_training(OUTPUT_DIR)


def _render_glossary() -> None:
    """Die Begriffe, ohne die keine Zahl auf dieser Seite lesbar ist."""
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
            "**Szenarien**: die Regelstrategie. *konstant/sofort* lädt ohne "
            "Rücksicht, *preisorientiert* wartet auf günstigen Strom, "
            "*eigennütziges RL* ist die gelernte Strategie."
        )


def main() -> None:
    st.set_page_config(page_title="GridKIT Dashboard", layout="wide")
    st.title("GridKIT Dashboard")
    st.caption(
        "Jeder Haushalt betreibt drei steuerbare Geräte-Agenten (EV, Batterie, Wärmepumpe) "
        "sowie eine exogene Dach-PV-Anlage, auf Basis realer, wetterabhängiger Profile "
        "(GridCreator/pyCity). Der Mechanismus wird unter vereinfachten Annahmen gezeigt "
        "(Ersatz-Lastfluss). Ein relativer Vergleich, keine Prognose."
    )
    summary = _load("summary.json")
    timelines = _load("timelines.json")
    if not summary:
        st.warning(f"Keine Ergebnisse in `{OUTPUT_DIR}/`. Zuerst `python -m GridKIT.scripts.run_experiment` ausführen.")
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
    # `streamlit run` importiert das Modul (name != __main__); nur dann automatisch
    # ausführen, NICHT bei einem einfachen Import (z. B. wenn die Web-App render_results importiert).
    main()
