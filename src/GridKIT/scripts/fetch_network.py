# scripts/fetch_network.py
# ─────────────────────────────────────────────────────────────
# Build a GridNetwork from real OpenStreetMap data via map_ui and save it
# to JSON, so the experiment can train on a real Karlsruhe feeder instead
# of the stub:  python -m GridKIT.scripts.run_experiment --network data/karlsruhe.json
#
# Needs network access (Overpass API).
#
# Usage:  python -m GridKIT.scripts.fetch_network \
#             --south 49.000 --west 8.400 --north 49.010 --east 8.415 \
#             --out data/karlsruhe.json
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _setup_paths() -> None:
    script_dir = Path(__file__).resolve().parent.parent
    src_dir = script_dir.parent
    for p in (str(src_dir), str(script_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(src_dir.parent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch an OSM area → GridNetwork JSON")
    # default: a small central-Karlsruhe box
    parser.add_argument("--south", type=float, default=49.000)
    parser.add_argument("--west", type=float, default=8.400)
    parser.add_argument("--north", type=float, default=49.010)
    parser.add_argument("--east", type=float, default=8.415)
    parser.add_argument("--out", type=str, default="data/karlsruhe.json")
    args = parser.parse_args()

    _setup_paths()
    from map_ui import AreaBounds, build_grid_network_from_bounds

    bounds = AreaBounds(south=args.south, west=args.west, north=args.north, east=args.east)
    print(f"Fetching OSM area {bounds.as_overpass_bbox()} (~{bounds.approx_area_km2():.2f} km²) …")
    try:
        result = build_grid_network_from_bounds(bounds)
    except Exception as e:  # noqa: BLE001 — network/Overpass failures are expected offline
        print(f"OSM fetch failed ({type(e).__name__}: {e}).")
        print("Check connectivity / Overpass availability, or use the bundled feeder "
              "(--network data/feeder_20.json).")
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save_grid_json(out)
    print(f"Saved {result.household_count} households / {result.bus_count} buses → {out}")


if __name__ == "__main__":
    main()
