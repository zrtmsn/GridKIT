# ─────────────────────────────────────────────────────────────
# map_ui/map_widget_training.py
#
# Save and training section for map_widget.py.
#
# This module stores the generated network together with the current
# household configuration and can start a training run.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

import streamlit as st


ACTION_COLUMN_COUNT = 2

NETWORK_SOURCE_MAP_UI = "map_ui"
RUNS_DIRECTORY_NAME = "runs"
TRAINING_LOG_FILENAME = "train.log"
TRAINING_MODULE = "GridKIT.scripts.train_run"


def show_training_section(network, household_configuration: dict[str, Any]) -> None:
    """
    Show the save and training controls for the current configuration.

    Training progress and results are shown in the dashboard app. This section
    only saves configurations and starts new training runs.
    """
    import core.constants as const
    from core import run_store as rs

    st.subheader("Speichern & Training")

    saved_run_id = st.session_state.pop("last_saved_run_id", None)
    if saved_run_id:
        st.success(f"Gespeichert als **{saved_run_id}**.")

    launched_run_id = st.session_state.pop("last_launched_run_id", None)
    if launched_run_id:
        st.success(
            f"Training für **{launched_run_id}** gestartet. Fortschritt und Ergebnisse siehst du im "
            f"Dashboard (`streamlit run src/GridKIT/dashboard/app.py`)."
        )

    st.caption(
        "Das ausgewählte Netz wird zusammen mit der aktuellen Haushaltskonfiguration gespeichert. "
        "Dadurch können mehrere Szenarien unabhängig voneinander abgelegt und später weiterverwendet werden."
    )

    n_households = len(getattr(network, "household_bus_ids", []))
    default_name = f"{getattr(network, 'area_name', None) or network.network_id} ({n_households} Haushalte)"

    run_name = st.text_input(
        "Name der Netzwerkkonfiguration",
        value=default_name,
        help="Nur ein Anzeigename, um mehrere gespeicherte Netze auseinanderzuhalten — muss nicht eindeutig sein.",
    )

    # Avoid starting multiple expensive training processes at the same time.
    active_runs = [
        run for run in rs.list_runs()
        if run["state"] == rs.RUNNING and not run.get("stale")
    ]

    if active_runs:
        names = ", ".join(f"**{run.get('name') or run['run_id']}**" for run in active_runs)
        st.warning(
            f"Es läuft bereits ein Training ({names}). Mehrere gleichzeitige Trainings können den "
            "Rechner überlasten und Abstürze verursachen — bitte warten, bis es fertig ist "
            "(Fortschritt im Dashboard), bevor ein weiteres gestartet wird. Speichern allein ist weiterhin möglich."
        )

    col1, col2 = st.columns(ACTION_COLUMN_COUNT)
    save_only_clicked = col1.button("Nur speichern")
    save_and_train_clicked = col2.button(
        "Speichern & Training starten",
        type="primary",
        disabled=bool(active_runs),
    )

    if save_only_clicked:
        run_id = rs.save_network_only(
            run_name,
            network,
            household_configuration,
            network_source=NETWORK_SOURCE_MAP_UI,
        )
        st.session_state["last_saved_run_id"] = run_id
        st.rerun()

    if save_and_train_clicked:
        run_id = rs.create_run(
            run_name,
            network,
            household_configuration,
            iterations=const.PIPELINE_MAX_TRAINING_ITERATIONS,
            seeds=const.PIPELINE_EVALUATION_SEEDS,
            network_source=NETWORK_SOURCE_MAP_UI,
        )

        try:
            _launch_training(run_id)
        except Exception as exc:
            rs.set_status(run_id, state=rs.FAILED, message=f"Start fehlgeschlagen: {exc}")
            st.error("Training konnte nicht gestartet werden.")
            st.exception(exc)
        else:
            st.session_state["last_launched_run_id"] = run_id
            st.rerun()


def _launch_training(run_id: str) -> None:
    """
    Start a detached training subprocess for one saved run.

    The subprocess writes its output to runs/<run_id>/train.log.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    from core import run_store as rs

    script_dir = Path(__file__).resolve().parent.parent
    src_dir = script_dir.parent
    repo_root = src_dir.parent

    run_directory = rs.run_dir(run_id, root=repo_root / RUNS_DIRECTORY_NAME)
    env = {
        **os.environ,
        "PYTHONPATH": f"{script_dir}{os.pathsep}{src_dir}",
        "PYTHONIOENCODING": "utf-8",  # prevents a Windows crash when training prints symbols
    }

    with open(run_directory / TRAINING_LOG_FILENAME, "w", encoding="utf-8") as logf:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                TRAINING_MODULE,
                "--run-dir",
                str(run_directory),
            ],
            cwd=str(repo_root),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )