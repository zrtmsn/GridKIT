# scripts/run_store.py
# ─────────────────────────────────────────────────────────────
# Registry + persistence for training runs, so the web app can launch real
# trainings in the background and browse previous ones (config, status, results).
#
# Layout on disk (default root: ./runs):
#   runs/<run_id>/
#     config.json      — name, params (iterations/seeds), network source, timestamps
#     network.json     — the GridNetwork trained on
#     layout.json      — the per-household device layout (HouseholdDevices list)
#     status.json      — {state, progress, iteration, total_iters, message, updated}
#     summary.json     — per-scenario metrics (written when eval finishes)
#     timelines.json   — representative episodes
#     checkpoints/     — trained per-device policies
#
# Filesystem-only + a configurable root → unit-testable without Streamlit/Ray.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.models import GridNetwork, HouseholdDevices

DEFAULT_ROOT = "runs"

# status states
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Sortable, filesystem-safe id: UTC timestamp to the second."""
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def run_dir(run_id: str, root: str | Path = DEFAULT_ROOT) -> Path:
    return Path(root) / run_id


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── creation ─────────────────────────────────────────────────
def create_run(
    name: str,
    network: GridNetwork,
    layout: dict[str, HouseholdDevices],
    *,
    iterations: int,
    seeds: int,
    network_source: str = "sample",
    run_id: str | None = None,
    root: str | Path = DEFAULT_ROOT,
) -> str:
    """Persist a new run (state=queued) and return its run_id."""
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
    (d / "network.json").write_text(network.model_dump_json(indent=2), encoding="utf-8")
    _write_json(d / "layout.json", {"households": [layout[b].model_dump() for b in sorted(layout)]})
    set_status(rid, state=QUEUED, progress=0.0, iteration=0, total_iters=iterations,
               message="queued", root=root)
    return rid


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
        out.append({**cfg,
                    "state": st.get("state", "unknown"),
                    "progress": st.get("progress", 0.0),
                    "iteration": st.get("iteration", 0),
                    "message": st.get("message", ""),
                    "has_results": (d / "summary.json").exists()})
    out.sort(key=lambda r: r.get("created", ""), reverse=True)
    return out


def load_config(run_id: str, root: str | Path = DEFAULT_ROOT) -> dict | None:
    return _read_json(run_dir(run_id, root) / "config.json")


def load_network(run_id: str, root: str | Path = DEFAULT_ROOT) -> GridNetwork:
    return GridNetwork.model_validate_json((run_dir(run_id, root) / "network.json").read_text())


def load_layout(run_id: str, root: str | Path = DEFAULT_ROOT) -> dict[str, HouseholdDevices]:
    data = _read_json(run_dir(run_id, root) / "layout.json") or {"households": []}
    hds = [HouseholdDevices.model_validate(h) for h in data["households"]]
    return {hd.bus_id: hd for hd in hds}


def load_results(run_id: str, root: str | Path = DEFAULT_ROOT) -> tuple[Any | None, Any | None]:
    d = run_dir(run_id, root)
    return _read_json(d / "summary.json"), _read_json(d / "timelines.json")
