# grid_model/gridcreator_loader.py
# ─────────────────────────────────────────────────────────────
# Import a GridCreator / ding0 grid as a core.GridNetwork.
#
# Two entry points:
#   load_gridcreator_network(nc_path)      — a saved GridCreator PyPSA .nc
#   build_grid_network_from_ding0(...)     — extract the LV grid covering an
#                                            ARBITRARY bbox straight out of the
#                                            ding0 grid archive (== GridCreator
#                                            step 1), so an area drawn on the map
#                                            gets REAL, individually-sized
#                                            transformers instead of the single
#                                            synthetic 160 kVA one that OSM
#                                            tagging can supply.
#
# ding0 produces a realistic LV grid: a forest of feeders, each rooted at one or
# more MV/LV transformers sized to its local load — exactly the multi-transformer
# topology our surrogate understands. Only the TOPOLOGY is used here; the time
# series in a reloaded .nc are NaN (a PyPSA-reload limitation) and step 1 carries
# none at all, so GridKIT drives the sim with its own pyCity device profiles.
#
# One bus-with-load = one household connection point (GridCreator may model
# several dwellings per bus; that finer granularity is not reconstructed).
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from core.models import BusModel, GridNetwork, LineModel, TransformerModel
from grid_model.builder import assign_clean_ids

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_GRIDCREATOR = _REPO_ROOT / "vendor" / "GridCreator"

#: Where GridCreator expects the ding0 grid archive (README: unpack input.zip
#: into the GridCreator repo). Override with $GRIDKIT_DING0_GRIDS_DIR.
DEFAULT_DING0_GRIDS_DIR = _VENDOR_GRIDCREATOR / "input" / "grids"

_ZENODO_URL = "https://zenodo.org/records/17884917"


class Ding0Unavailable(RuntimeError):
    """Base: a ding0-backed grid could not be produced for the request."""


class Ding0DataMissing(Ding0Unavailable):
    """The ding0 grid archive itself is absent — a setup problem, not a map problem."""


class Ding0AreaNotCovered(Ding0Unavailable):
    """The archive is present but holds no grid for the requested bbox."""


# ══════════════════════════════════════════════════════════════
# pypsa.Network → core.GridNetwork
# ══════════════════════════════════════════════════════════════
def _num(row, name: str, default: float) -> float:
    """Column value as float, tolerating a missing column or NaN.

    ding0 topology CSVs and a fully-processed GridCreator .nc do not carry
    exactly the same columns, so every electrical attribute is read defensively.
    """
    import pandas as pd

    try:
        value = row.get(name)
    except AttributeError:  # not a Series-like row
        value = getattr(row, name, None)
    if value is None or not pd.notna(value):
        return default
    return float(value)


def _network_from_pypsa(n, network_id: str, residential_only: bool) -> GridNetwork:
    """Convert an in-memory PyPSA network (topology only) to a GridNetwork."""
    import pandas as pd

    v_nom = {str(b): float(r.v_nom) for b, r in n.buses.iterrows()}

    buses: list[BusModel] = []
    for bid, row in n.buses.iterrows():
        x = float(row.x) if "x" in row and pd.notna(row.x) else None
        y = float(row.y) if "y" in row and pd.notna(row.y) else None
        buses.append(BusModel(bus_id=str(bid), v_nom_kv=float(row.v_nom), x_coord=x, y_coord=y))

    lines: list[LineModel] = []
    for lid, row in n.lines.iterrows():
        length = _num(row, "length", 0.0)
        if length <= 0:
            length = 0.03
        r = _num(row, "r", 0.0)
        x = _num(row, "x", 0.0)
        s_nom = _num(row, "s_nom", 0.0)
        v = v_nom.get(str(row.bus0), 0.4)
        i_max = s_nom / (v * math.sqrt(3)) if (s_nom > 0 and v > 0) else 0.2   # preserve s_nom in the surrogate
        lines.append(LineModel(
            line_id=str(lid), from_bus=str(row.bus0), to_bus=str(row.bus1),
            length_km=length,
            r_ohm_per_km=(r / length) if r > 0 else 0.3,
            x_ohm_per_km=(x / length) if x > 0 else 0.08,
            max_i_ka=max(i_max, 1e-3),
        ))

    transformers: list[TransformerModel] = []
    for tid, row in n.transformers.iterrows():
        b0, b1 = str(row.bus0), str(row.bus1)
        hv, lv = (b0, b1) if v_nom.get(b0, 20.0) >= v_nom.get(b1, 0.4) else (b1, b0)  # LV = lower voltage = feeder root
        s_nom = _num(row, "s_nom", 0.0)
        transformers.append(TransformerModel(
            trafo_id=str(tid), hv_bus=hv, lv_bus=lv, s_nom_mva=max(s_nom, 1e-3),
            vn_hv_kv=v_nom.get(hv, 20.0), vn_lv_kv=v_nom.get(lv, 0.4),
        ))

    # household connection points = buses carrying a (residential) load
    if len(n.loads):
        if residential_only and "type" in n.loads.columns:
            mask = n.loads["type"].astype(str).str.contains("conventional", case=False, na=False)
            load_buses = n.loads.loc[mask, "bus"] if mask.any() else n.loads["bus"]
        else:
            load_buses = n.loads["bus"]
        wanted = sorted({str(b) for b in load_buses})
    else:
        wanted = []
    bus_ids = {b.bus_id for b in buses}
    household_bus_ids = [b for b in wanted if b in bus_ids]

    buses, lines, transformers, household_bus_ids, _, _, _ = assign_clean_ids(
        buses, lines, transformers, household_bus_ids,
    )

    return GridNetwork(
        network_id=network_id,
        buses=buses, lines=lines, transformers=transformers,
        household_bus_ids=household_bus_ids,
    )


def _quiet_pypsa():
    """Silence PyPSA/ding0 import chatter (they log heavily per component)."""
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)


def load_gridcreator_network(
    nc_path: str | Path,
    *,
    network_id: str | None = None,
    residential_only: bool = True,
) -> GridNetwork:
    """Build a GridNetwork from a GridCreator PyPSA .nc (topology only)."""
    _quiet_pypsa()
    import pypsa

    n = pypsa.Network(str(nc_path))
    return _network_from_pypsa(n, network_id or Path(nc_path).stem, residential_only)


# ══════════════════════════════════════════════════════════════
# bbox → ding0 LV grid  (GridCreator step 1)
# ══════════════════════════════════════════════════════════════
def resolve_grids_dir(grids_dir: str | Path | None = None) -> Path:
    """Locate the ding0 grid archive: argument → env var → vendored default."""
    if grids_dir is not None:
        return Path(grids_dir)
    env = os.environ.get("GRIDKIT_DING0_GRIDS_DIR")
    return Path(env) if env else DEFAULT_DING0_GRIDS_DIR


def ding0_archive_available(grids_dir: str | Path | None = None) -> bool:
    """True when the archive holds at least one ding0 grid we can read."""
    root = resolve_grids_dir(grids_dir)
    if not root.is_dir():
        return False
    return any((sub / "topology" / "buses.csv").is_file() for sub in root.iterdir() if sub.is_dir())


def _import_ding0_generator():
    """Import GridCreator's ding0 extractor from the vendored checkout.

    Only `ding0_grid_generator` is imported — deliberately NOT `main_functions`,
    which drags in weather/CDS-API machinery we have no use for. Its dependencies
    (pypsa, networkx, pandas, geopandas, tqdm) are already GridKIT dependencies.
    """
    if not (_VENDOR_GRIDCREATOR / "ding0_grid_generator.py").is_file():
        raise Ding0DataMissing(
            f"GridCreator checkout not found at {_VENDOR_GRIDCREATOR}. "
            "Initialise the vendor submodule first."
        )
    path = str(_VENDOR_GRIDCREATOR)
    if path not in sys.path:
        sys.path.insert(0, path)
    import ding0_grid_generator  # noqa: E402  (vendored, path-injected)

    return ding0_grid_generator


def _bbox_from_buses(net) -> list[float]:
    """[min_x, min_y, max_x, max_y] over a network's bus coordinates.

    Mirrors GridCreator's `functions.compute_bbox_from_buses`, reimplemented here
    so we don't import that heavyweight module for four lines of arithmetic.
    """
    b = net.buses
    return [float(b["x"].min()), float(b["y"].min()), float(b["x"].max()), float(b["y"].max())]


def build_grid_network_from_ding0(
    south: float,
    west: float,
    north: float,
    east: float,
    *,
    grids_dir: str | Path | None = None,
    network_id: str | None = None,
    residential_only: bool = True,
) -> GridNetwork:
    """Extract the real ding0 LV grid covering a bbox, as a GridNetwork.

    Plain lat/lon floats rather than a map_ui type on purpose: `grid_model` must
    not import another feature module (only `core`). The caller converts.

    Reproduces GridCreator step 1 (`main_functions.ding0_grid`): pick every ding0
    bus inside the box, walk each one to its nearest LV transformer so feeders
    come out whole, then re-extract over the resulting extent to pull in the rest
    of the touched feeders.

    Raises `Ding0DataMissing` when the archive is absent and
    `Ding0AreaNotCovered` when it holds nothing for this box.
    """
    root = resolve_grids_dir(grids_dir)
    if not ding0_archive_available(root):
        raise Ding0DataMissing(
            f"No ding0 grid archive at {root}. Download input.zip from {_ZENODO_URL}, "
            f"unpack it, and place its 'grids' folder there (or point "
            f"$GRIDKIT_DING0_GRIDS_DIR at it)."
        )

    _quiet_pypsa()
    ding0 = _import_ding0_generator()

    # ding0 CSVs use x = longitude, y = latitude → (min_x, min_y, max_x, max_y)
    bbox = [west, south, east, north]
    grid = ding0.load_grid(bbox, str(root))
    if grid.buses.empty:
        raise Ding0AreaNotCovered(
            "No ding0 grid covers this area. The archive holds German LV grids only, "
            "and only the districts you downloaded — try an area inside one of them."
        )

    # widen to the full extent of the feeders we touched, then take them whole
    grid = ding0.load_grid(_bbox_from_buses(grid), str(root))
    if grid.buses.empty:  # pragma: no cover (first pass succeeded, so this cannot normally happen)
        raise Ding0AreaNotCovered("ding0 extraction returned an empty grid for this area.")

    default_id = f"ding0_{south:.4f}_{west:.4f}_{north:.4f}_{east:.4f}"
    return _network_from_pypsa(grid, network_id or default_id, residential_only)
