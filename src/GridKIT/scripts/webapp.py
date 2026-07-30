# scripts/webapp.py
# ─────────────────────────────────────────────────────────────
# GridKIT — unified web app: Design & Train  ·  Runs (history)  ·  Results.
#
# Workflow: choose an area / grid → customise devices → launch REAL training in
# the BACKGROUND (a detached subprocess) → keep using the app (browse previous
# runs and their results) while it trains.
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


# ══════════════════════════════════════════════════════════════
# Background training launch
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
# Pages
# ══════════════════════════════════════════════════════════════
def page_design() -> None:
    st.title("Design & Train")
    render_designer()   # map → grid → device sliders + click-to-edit → quick baseline preview

    ss = st.session_state
    st.divider()
    st.header("🚀 Launch real training (background)")
    if not ss.get("network"):
        st.info("Build or load a network above first.")
        return
    n_homes = len(ss.network.household_bus_ids)
    name = st.text_input("Run name", value=f"{n_homes} homes")
    c1, c2 = st.columns(2)
    iters = int(c1.number_input("Training iterations", 5, 300, 30, step=5))
    sds = int(c2.number_input("Evaluation seeds", 1, 16, 6))
    st.caption("Trains an IPPO policy on THIS grid + layout, then evaluates baselines + RL. "
               "Runs in the background — switch to the Runs tab and keep working while it trains.")
    if st.button("Launch training in background", type="primary"):
        rid = run_store.create_run(name, ss.network, ss.layout, iterations=iters, seeds=sds,
                                   network_source=ss.get("source_kind", "designed"), root=RUNS_ROOT)
        launch_training(rid)
        ss.selected_run = rid
        st.success(f"Launched **{rid}**. Track progress in the **Runs** tab — the app stays usable.")


def page_runs() -> None:
    st.title("Runs")
    runs = run_store.list_runs(root=RUNS_ROOT)
    if st.button("🔄 Refresh"):
        st.rerun()
    if not runs:
        st.info("No training runs yet. Launch one from **Design & Train**.")
        return

    st.dataframe(
        [{"run": r["run_id"], "name": r.get("name", ""), "state": r["state"],
          "progress": f"{r.get('progress', 0) * 100:.0f}%", "homes": r.get("n_households", 0),
          "iters": r.get("iterations", 0), "created": r.get("created", ""),
          "results": "✓" if r.get("has_results") else "—"} for r in runs],
        width="stretch", hide_index=True,
    )

    ss = st.session_state
    ids = [r["run_id"] for r in runs]
    by_id = {r["run_id"]: r for r in runs}
    default = ids.index(ss.selected_run) if ss.get("selected_run") in ids else 0
    sel = st.selectbox("Select a run", ids, index=default,
                       format_func=lambda i: f"{by_id[i].get('name','')} · {by_id[i]['state']} ({i})")
    ss.selected_run = sel

    r = by_id[sel]
    state = r["state"]
    if state == run_store.RUNNING:
        st.progress(min(1.0, r.get("progress", 0.0)), text=r.get("message", "training…"))
        st.caption("Still training — hit Refresh to update, or open **Results** when it's done.")
    elif state == run_store.DONE:
        st.success(f"Done — open the **Results** tab. {r.get('message','')}")
    elif state == run_store.FAILED:
        st.error(f"Failed: {r.get('message','')}  (see runs/{sel}/train.log)")
    else:
        st.info(f"State: {state}")


def page_results() -> None:
    st.title("Results")
    ss = st.session_state

    runs = run_store.list_runs(root=RUNS_ROOT)
    if not runs:
        st.info("No runs yet — launch one from **Design & Train**.")
        return

    # choose the exact run to view, right here (syncs with the Runs tab)
    ids = [r["run_id"] for r in runs]
    by_id = {r["run_id"]: r for r in runs}
    default = ids.index(ss.selected_run) if ss.get("selected_run") in ids else 0
    sel = st.selectbox(
        "Run to view", ids, index=default,
        format_func=lambda i: (f"{by_id[i].get('name','')} · {by_id[i]['state']}"
                               f"{'  ✓ results' if by_id[i].get('has_results') else '  (no results yet)'}  ·  {i}"),
    )
    ss.selected_run = sel

    cfg = run_store.load_config(sel, root=RUNS_ROOT) or {}
    st.caption(f"Run **{sel}** — {cfg.get('name','')} · {cfg.get('n_households','?')} homes · "
               f"{cfg.get('iterations','?')} iters · {cfg.get('seeds','?')} seeds")
    summary, timelines = run_store.load_results(sel, root=RUNS_ROOT)
    if not summary:
        st.warning("No results yet for this run — it may still be training or have failed (check the **Runs** tab).")
        return
    # the run's own stored topology, so the overload map shows the grid that was trained on
    try:
        network = run_store.load_network(sel, root=RUNS_ROOT)
    except Exception:
        network = None
    render_results(summary, timelines, network=network)


def main() -> None:
    st.set_page_config(page_title="GridKIT", layout="wide")
    st.session_state.setdefault("selected_run", None)
    nav = st.navigation({
        "Design": [st.Page(page_design, title="Design & Train", icon="🏘️")],
        "History": [st.Page(page_runs, title="Runs", icon="📋"),
                    st.Page(page_results, title="Results", icon="📊")],
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
