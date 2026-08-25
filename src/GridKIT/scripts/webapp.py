# scripts/webapp.py
# ─────────────────────────────────────────────────────────────
# GridKIT — unified web app: Design & Train  ·  Runs (history)  ·  Results.
#
# Workflow: choose an area / grid → customise devices → launch REAL training in
# the BACKGROUND (a detached subprocess) → keep using the app (browse previous
# runs and their results) while it trains.
#
# UI text is German. Tab names used in this file: "Entwurf & Training"
# (Design & Train), "Läufe" (Runs), "Ergebnisse" (Results) — kept consistent
# between st.navigation() and every caption/message that refers to a tab by
# name. run_store's RUNNING/DONE/FAILED constants stay in English — they're
# persisted state values compared against status.json on disk, not display
# text; only their on-screen labels are translated (STATE_LABELS_DE below).
#
# Run:  PYTHONPATH=src streamlit run src/GridKIT/scripts/webapp.py
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ── path bootstrap ──
_SCRIPT_DIR = Path(__file__).resolve().parent.parent   # src/GridKIT/
_SRC_DIR = _SCRIPT_DIR.parent                           # src/
_REPO_ROOT = _SRC_DIR.parent                            # repo root
for _p in (str(_SCRIPT_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNS_ROOT = _REPO_ROOT / "runs"

import streamlit as st

from scripts import run_store
from scripts.grid_designer import render_designer
from GridKIT.dashboard.app import render_results

STATE_LABELS_DE = {
    run_store.RUNNING: "läuft",
    run_store.DONE: "fertig",
    run_store.FAILED: "fehlgeschlagen",
}


# ══════════════════════════════════════════════════════════════
# Hintergrund-Trainingsstart
# ══════════════════════════════════════════════════════════════
def launch_training(run_id: str) -> None:
    """Spawn a detached training subprocess writing into runs/<run_id>/."""
    d = run_store.run_dir(run_id, root=RUNS_ROOT)
    env = {**os.environ, "PYTHONPATH": f"{_SCRIPT_DIR}{os.pathsep}{_SRC_DIR}"}
    logf = open(d / "train.log", "w")
    subprocess.Popen(
        [sys.executable, "-m", "GridKIT.scripts.train_run", "--run-dir", str(d)],
        cwd=str(_REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT,
    )


# ══════════════════════════════════════════════════════════════
# Seiten
# ══════════════════════════════════════════════════════════════
def page_design() -> None:
    st.title("Entwurf & Training")
    render_designer()   # Karte → Netz → Geräte-Regler + Klick-Bearbeitung → schnelle Basis-Vorschau

    ss = st.session_state
    st.divider()
    st.header("🚀 Echtes Training starten (Hintergrund)")
    if not ss.get("network"):
        st.info("Zuerst oben ein Netz erstellen oder laden.")
        return
    n_homes = len(ss.network.household_bus_ids)
    name = st.text_input("Name des Laufs", value=f"{n_homes} Haushalte")
    c1, c2 = st.columns(2)
    iters = int(c1.number_input("Trainingsiterationen", 5, 300, 30, step=5))
    sds = int(c2.number_input("Evaluierungs-Seeds", 1, 16, 6))
    st.caption("Trainiert eine IPPO-Policy auf DIESEM Netz + Layout und evaluiert dann die "
               "Baselines + RL. Läuft im Hintergrund — zum Tab **Läufe** wechseln und "
               "währenddessen weiterarbeiten.")
    if st.button("Training im Hintergrund starten", type="primary"):
        rid = run_store.create_run(name, ss.network, ss.layout, iterations=iters, seeds=sds,
                                   network_source=ss.get("source_kind", "designed"), root=RUNS_ROOT)
        launch_training(rid)
        ss.selected_run = rid
        st.success(f"**{rid}** gestartet. Fortschritt im Tab **Läufe** verfolgen — die App bleibt währenddessen nutzbar.")


def page_runs() -> None:
    st.title("Läufe")
    runs = run_store.list_runs(root=RUNS_ROOT)
    if st.button("🔄 Aktualisieren"):
        st.rerun()
    if not runs:
        st.info("Noch keine Trainingsläufe. Einen unter **Entwurf & Training** starten.")
        return

    st.dataframe(
        [{"Lauf": r["run_id"], "Name": r.get("name", ""), "Status": STATE_LABELS_DE.get(r["state"], r["state"]),
          "Fortschritt": f"{r.get('progress', 0) * 100:.0f}%", "Haushalte": r.get("n_households", 0),
          "Iterationen": r.get("iterations", 0), "Erstellt": r.get("created", ""),
          "Ergebnisse": "✓" if r.get("has_results") else "—"} for r in runs],
        width="stretch", hide_index=True,
    )

    ss = st.session_state
    ids = [r["run_id"] for r in runs]
    by_id = {r["run_id"]: r for r in runs}
    default = ids.index(ss.selected_run) if ss.get("selected_run") in ids else 0
    sel = st.selectbox("Lauf auswählen", ids, index=default,
                       format_func=lambda i: f"{by_id[i].get('name','')} · {STATE_LABELS_DE.get(by_id[i]['state'], by_id[i]['state'])} ({i})")
    ss.selected_run = sel

    r = by_id[sel]
    state = r["state"]
    if state == run_store.RUNNING:
        st.progress(min(1.0, r.get("progress", 0.0)), text=r.get("message", "läuft…"))
        st.caption("Noch am Trainieren — Aktualisieren für den neuesten Stand, oder **Ergebnisse** öffnen, sobald fertig.")
    elif state == run_store.DONE:
        st.success(f"Fertig — Tab **Ergebnisse** öffnen. {r.get('message','')}")
    elif state == run_store.FAILED:
        st.error(f"Fehlgeschlagen: {r.get('message','')}  (siehe runs/{sel}/train.log)")
    else:
        st.info(f"Status: {STATE_LABELS_DE.get(state, state)}")


def page_results() -> None:
    st.title("Ergebnisse")
    ss = st.session_state

    runs = run_store.list_runs(root=RUNS_ROOT)
    if not runs:
        st.info("Noch keine Läufe — einen unter **Entwurf & Training** starten.")
        return

    # genau den anzuzeigenden Lauf hier auswählen (synchron mit dem Tab „Läufe")
    ids = [r["run_id"] for r in runs]
    by_id = {r["run_id"]: r for r in runs}
    default = ids.index(ss.selected_run) if ss.get("selected_run") in ids else 0
    sel = st.selectbox(
        "Anzuzeigender Lauf", ids, index=default,
        format_func=lambda i: (f"{by_id[i].get('name','')} · {STATE_LABELS_DE.get(by_id[i]['state'], by_id[i]['state'])}"
                               f"{'  ✓ Ergebnisse' if by_id[i].get('has_results') else '  (noch keine Ergebnisse)'}  ·  {i}"),
    )
    ss.selected_run = sel

    cfg = run_store.load_config(sel, root=RUNS_ROOT) or {}
    st.caption(f"Lauf **{sel}** — {cfg.get('name','')} · {cfg.get('n_households','?')} Haushalte · "
               f"{cfg.get('iterations','?')} Iterationen · {cfg.get('seeds','?')} Seeds")
    summary, timelines = run_store.load_results(sel, root=RUNS_ROOT)
    if not summary:
        st.warning("Für diesen Lauf liegen noch keine Ergebnisse vor — er trainiert eventuell noch oder ist fehlgeschlagen (Tab **Läufe** prüfen).")
        return
    # die im Lauf gespeicherte Topologie, damit die Überlastungskarte das trainierte Netz zeigt
    try:
        network = run_store.load_network(sel, root=RUNS_ROOT)
    except Exception:
        network = None
    render_results(summary, timelines, network=network)


def main() -> None:
    st.set_page_config(page_title="GridKIT", layout="wide")
    st.session_state.setdefault("selected_run", None)
    nav = st.navigation({
        "Entwurf": [st.Page(page_design, title="Entwurf & Training", icon="🏘️")],
        "Verlauf": [st.Page(page_runs, title="Läufe", icon="📋"),
                    st.Page(page_results, title="Ergebnisse", icon="📊")],
    })
    nav.run()


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    main()
elif _running_under_streamlit():
    main()
