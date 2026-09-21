# scripts/fetch_network.py
# ─────────────────────────────────────────────────────────────
# Build a GridNetwork from the same grid_model/GridCreator path used by map_ui
# and save it to JSON.
#
#
# Usage:
#   python -m GridKIT.scripts.fetch_network \
#       --south 49.000 --west 8.400 --north 49.010 --east 8.415 \
#       --out data/karlsruhe.json
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_GRIDCREATOR_CONDA_ENV = "GridCreator"


def _setup_paths() -> None:
    script_dir = Path(__file__).resolve().parent.parent
    src_dir = script_dir.parent

    for path in (str(src_dir), str(script_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.chdir(src_dir.parent)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a GridNetwork with grid_model/GridCreator and save it as JSON."
    )

    # Default: a small central-Karlsruhe box.
    parser.add_argument("--south", type=float, default=49.000)
    parser.add_argument("--west", type=float, default=8.400)
    parser.add_argument("--north", type=float, default=49.010)
    parser.add_argument("--east", type=float, default=8.415)

    parser.add_argument("--out", type=str, default="data/karlsruhe.json")
    parser.add_argument("--scenario", type=str, default="fetched_network")
    parser.add_argument("--conda-env", type=str, default=DEFAULT_GRIDCREATOR_CONDA_ENV)

    args = parser.parse_args()

    _setup_paths()

    from grid_model.builder import OSMNetworkBuilder
    from map_ui.area_bounds import AreaBounds

    bounds = AreaBounds(
        south=args.south,
        west=args.west,
        north=args.north,
        east=args.east,
    )

    print(
        "Building GridNetwork with grid_model/GridCreator "
        f"for bbox {bounds.as_overpass_bbox} "
        f"(~{bounds.approx_area_km2():.2f} km²) ..."
    )

    try:
        builder = OSMNetworkBuilder(
            top=bounds.north,
            bottom=bounds.south,
            left=bounds.west,
            right=bounds.east,
            scenario=args.scenario,
            conda_env=args.conda_env,
        )

        network = builder.build()

    except Exception as exc:  # noqa: BLE001
        print(f"GridNetwork build failed ({type(exc).__name__}: {exc}).")
        print(
            "Please check the selected coordinates, the grid_model/GridCreator setup, "
            "and the configured Conda environment."
        )
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(network.model_dump_json(indent=2), encoding="utf-8")

    household_count = len(getattr(network, "household_bus_ids", []) or [])
    bus_count = len(getattr(network, "buses", []) or [])
    line_count = len(getattr(network, "lines", []) or [])
    transformer_count = len(getattr(network, "transformers", []) or [])

    print(
        f"Saved GridNetwork → {out}\n"
        f"Network ID: {network.network_id}\n"
        f"Buses: {bus_count}\n"
        f"Lines: {line_count}\n"
        f"Transformers: {transformer_count}\n"
        f"Households: {household_count}"
    )


if __name__ == "__main__":
    main()