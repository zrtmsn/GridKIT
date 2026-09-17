# core/run_store.py
# ─────────────────────────────────────────────────────────────
# Registry + persistence for saved networks and their training runs, so the
# UI can save what it builds and later browse every run's results.
#
# Layout on disk (default root: ./runs):
#   runs/<run_id>/
#     config.json                  — name, iterations/seeds, network source, timestamps
#     grid_network.json            — the GridNetwork this run is for (same shape
#                                     map_ui's own "grid_network.json herunterladen"
#                                     button already produces — GridNetwork.model_dump_json())
#     household_configuration.json — the household config dict (same shape
#                                     map_ui's own "household_configuration.json
#                                     herunterladen" button already produces —
#                                     see map_ui/household_config.py). Stored
#                                     verbatim, no conversion to
#                                     core.models.HouseholdDevices here — that's
#                                     a training script's job, not the store's.
#     status.json                  — {state, progress, iteration, total_iters, message, updated}
#     summary.json                 — per-scenario metrics (written when eval finishes)
#     timelines.json               — representative episodes
#     checkpoints/                 — trained per-device RL policies (a training
#                                     script's concern — this module only reserves
#                                     the path, run_dir(run_id) / "checkpoints")
#
# Lives in core (not scripts/grid_model/map_ui) so any module — map_ui to save
# what it builds, a future training script to update status/results, a future
# dashboard to browse them — can import it directly without inverting the
# module-boundary rule (core is the one thing every module already imports).
#
# Filesystem-only + a configurable root → unit-testable without Streamlit/Ray.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.models import GridNetwork

DEFAULT_ROOT = "runs"

# status states
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

# A "running" status this old with no update is almost certainly an orphaned
# process (killed by the OS — e.g. out-of-memory — rather than exiting
# normally), not real progress: every training iteration touches status.json,
# so a live run updates far more often than this. train_run.py catches
# ordinary exceptions and interruption and marks the run FAILED itself, but a
# SIGKILL can't be caught by anything running inside the killed process — this
# is the only way stale "running" entries get flagged at all.
STALE_AFTER_SECONDS = 10 * 60


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_stale(updated: str | None) -> bool:
    if not updated:
        return False
    try:
        ts = datetime.fromisoformat(updated)
    except ValueError:
        return False
    return (datetime.now(timezone.utc) - ts).total_seconds() > STALE_AFTER_SECONDS


def new_run_id() -> str:
    """Sortable, filesystem-safe id: UTC timestamp to the second."""
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def run_dir(run_id: str, root: str | Path = DEFAULT_ROOT) -> Path:
    return Path(root) / run_id


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── creation ─────────────────────────────────────────────────
def create_run(
    name: str,
    network: GridNetwork,
    household_configuration: dict[str, Any],
    *,
    iterations: int = 0,
    seeds: int = 0,
    network_source: str = "map_ui",
    run_id: str | None = None,
    root: str | Path = DEFAULT_ROOT,
) -> str:
    """Persist a new run (state=queued) and return its run_id.

    `iterations`/`seeds` default to 0 — pass real values only once training is
    actually about to run. A network can be saved on its own, with no training
    attached yet (see save_network_only, a thin wrapper for exactly that case).
    """
    rid = run_id or new_run_id()
    d = run_dir(rid, root)
    d.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": rid,
        "name": name or rid,
        "created": _now_iso(),
        "network_source": network_source,
        "n_households": len(network.household_bus_ids),
        "iterations": iterations,
        "seeds": seeds,
    }
    _write_json(d / "config.json", config)
    (d / "grid_network.json").write_text(network.model_dump_json(indent=2), encoding="utf-8")
    _write_json(d / "household_configuration.json", household_configuration)
    set_status(rid, state=QUEUED, progress=0.0, iteration=0, total_iters=iterations,
               message="queued", root=root)
    return rid


def save_network_only(
    name: str,
    network: GridNetwork,
    household_configuration: dict[str, Any],
    *,
    network_source: str = "map_ui",
    run_id: str | None = None,
    root: str | Path = DEFAULT_ROOT,
) -> str:
    """Persist a network + household config with no training attached yet —
    for "just save what I built in the UI", independent of ever training it."""
    return create_run(name, network, household_configuration, iterations=0, seeds=0,
                       network_source=network_source, run_id=run_id, root=root)


# ── status ───────────────────────────────────────────────────
def set_status(run_id: str, *, root: str | Path = DEFAULT_ROOT, **fields: Any) -> dict:
    """Merge fields into status.json (stamps `updated`). Returns the new status."""
    path = run_dir(run_id, root) / "status.json"
    status = _read_json(path) or {}
    status.update(fields)
    status["updated"] = _now_iso()
    _write_json(path, status)
    return status


def get_status(run_id: str, root: str | Path = DEFAULT_ROOT) -> dict:
    return _read_json(run_dir(run_id, root) / "status.json") or {"state": "unknown"}


# ── listing / loading ────────────────────────────────────────
def list_runs(root: str | Path = DEFAULT_ROOT) -> list[dict]:
    """All runs, newest first, each a merged config + status summary dict."""
    base = Path(root)
    if not base.exists():
        return []
    out: list[dict] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        cfg = _read_json(d / "config.json")
        if cfg is None:
            continue
        st = _read_json(d / "status.json") or {}
        state = st.get("state", "unknown")
        out.append({**cfg,
                    "state": state,
                    "progress": st.get("progress", 0.0),
                    "iteration": st.get("iteration", 0),
                    "message": st.get("message", ""),
                    "has_results": (d / "summary.json").exists(),
                    # only meaningful while state == RUNNING; see STALE_AFTER_SECONDS
                    "stale": state == RUNNING and _is_stale(st.get("updated"))})
    # Tiebreak by run_id: `created` is wall-clock to the second, so two runs
    # made within the same second would otherwise tie and fall back to
    # non-deterministic filesystem iteration order.
    out.sort(key=lambda r: (r.get("created", ""), r.get("run_id", "")), reverse=True)
    return out


def load_config(run_id: str, root: str | Path = DEFAULT_ROOT) -> dict | None:
    return _read_json(run_dir(run_id, root) / "config.json")


def load_network(run_id: str, root: str | Path = DEFAULT_ROOT) -> GridNetwork:
    return GridNetwork.model_validate_json((run_dir(run_id, root) / "grid_network.json").read_text())


def load_household_configuration(run_id: str, root: str | Path = DEFAULT_ROOT) -> dict[str, Any] | None:
    return _read_json(run_dir(run_id, root) / "household_configuration.json")


def load_results(run_id: str, root: str | Path = DEFAULT_ROOT) -> tuple[Any | None, Any | None]:
    d = run_dir(run_id, root)
    summary = _read_json(d / "summary.json")
    timelines = _read_json(d / "timelines.json")
    if isinstance(timelines, dict):
        # train_run.py writes {scenario_label: timeline_dict} for a UI-launched
        # run; the dashboard (ueberblick/auslastung/geraete/vergleich) — built
        # against run_experiment.py's batch output — expects a plain list of
        # timeline dicts instead. Each dict already carries its own "scenario"/
        # "penetration" keys (see scripts.run_experiment._timeline), so this is
        # a lossless reshape, not a data change.
        timelines = list(timelines.values())
    return summary, timelines


def save_results(
    run_id: str,
    summary: list[dict[str, Any]],
    timelines: Any,
    *,
    root: str | Path = DEFAULT_ROOT,
) -> None:
    """Write summary.json/timelines.json for a run and mark it done."""
    d = run_dir(run_id, root)
    _write_json(d / "summary.json", summary)
    _write_json(d / "timelines.json", timelines)
    set_status(run_id, state=DONE, progress=1.0, message="done", root=root)


def delete_run(run_id: str, root: str | Path = DEFAULT_ROOT) -> None:
    """Permanently remove a run directory (network, config, results, checkpoints). No undo."""
    d = run_dir(run_id, root)
    if d.exists():
        shutil.rmtree(d)
