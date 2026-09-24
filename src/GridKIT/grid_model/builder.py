#grid_model/builder.py
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

import core.constants as const
from core import BusModel, FeederSummary, GridNetwork, LineModel, TransformerModel, settings
from core.models import HouseholdDevices
from core.protocols import NetworkBuilderProtocol

EV_BUS_SUFFIX = "_E_Car"


def _first_episode_day(series: pd.Series) -> list[float]:
    """
    GridCreator's own snapshots are hourly (a full year); GridKIT episodes run
    at EPISODE_STEPS steps of TIMESTEP_MINUTES each (96 x 15 min). Slice the
    first 24 h (one representative day, positionally — GridCreator's export
    isn't noon-aligned like device_profiles.py's annual pool) and linearly
    resample onto the episode clock; a straight positional slice only lines up
    1:1 when TIMESTEP_MINUTES == 60.
    """
    hourly = np.asarray(series.iloc[:24].tolist(), dtype=float)
    steps_per_hour = 60 / const.TIMESTEP_MINUTES
    xp = np.arange(24) * steps_per_hour
    x = np.arange(const.EPISODE_STEPS)
    return np.interp(x, xp, hourly).tolist()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def assign_clean_ids(
    buses: list[BusModel],
    lines: list[LineModel],
    transformers: list[TransformerModel],
    household_bus_ids: list[str],
    household_load_profile_kw: dict[str, list[float]] | None = None,
    ev_availability: dict[str, list[bool]] | None = None,
    household_devices: dict[str, HouseholdDevices] | None = None,
) -> tuple[list[BusModel], list[LineModel], list[TransformerModel], list[str],
           dict[str, list[float]], dict[str, list[bool]], dict[str, HouseholdDevices]]:
    """Replace GridCreator's/ding0's verbose internal ids (e.g.
    "BranchTee_mvgd_32359_lvgd_6762200004_building_5260160") with short,
    human-readable ones — "household_1", "transformer_1_lv", "bus_3",
    "line_7". These ids are shown directly to the user (map popups,
    household config, the dashboard's overload map) and are used as dict
    keys throughout the rest of the pipeline, so renaming once here, right
    at extraction, is enough to fix it everywhere downstream.

    Numbering is deterministic (alphabetical over the ORIGINAL ids), so the
    same GridCreator/ding0 output always gets the same clean names. A bus
    that is both a transformer's hv/lv bus and (rarely) also carries a
    household load keeps its transformer-derived name — the electrical role
    wins over the household numbering.
    """
    household_load_profile_kw = household_load_profile_kw or {}
    ev_availability = ev_availability or {}
    household_devices = household_devices or {}

    trafo_id_map: dict[str, str] = {}
    bus_id_map: dict[str, str] = {}

    # Group by lv_bus BEFORE numbering: two transformers can share one real
    # busbar (a ding0 "reinforced" parallel pair). Numbering raw rows 1..N
    # independently of that leaves gaps in the hv/lv bus names — the busbar
    # keeps whichever member's number claimed it first, so the other
    # member's own number never appears on any bus.
    groups: dict[str, list[TransformerModel]] = {}
    for trafo in transformers:
        groups.setdefault(trafo.lv_bus, []).append(trafo)

    for i, lv_bus in enumerate(sorted(groups, key=lambda b: min(t.trafo_id for t in groups[b])), start=1):
        members = sorted(groups[lv_bus], key=lambda t: t.trafo_id)
        bus_id_map[lv_bus] = f"transformer_{i}_lv"
        for member in members:
            bus_id_map.setdefault(member.hv_bus, f"transformer_{i}_hv")

        trafo_id_map[members[0].trafo_id] = f"transformer_{i}"
        for j, member in enumerate(members[1:], start=2):
            trafo_id_map[member.trafo_id] = (
                f"transformer_{i}_reinforced" if j == 2 else f"transformer_{i}_reinforced_{j}"
            )

    for i, bus_id in enumerate(sorted(household_bus_ids), start=1):
        bus_id_map.setdefault(bus_id, f"household_{i}")

    remaining = sorted(b.bus_id for b in buses if b.bus_id not in bus_id_map)
    for i, bus_id in enumerate(remaining, start=1):
        bus_id_map[bus_id] = f"bus_{i}"

    line_id_map = {
        ln.line_id: f"line_{i}"
        for i, ln in enumerate(sorted(lines, key=lambda l: l.line_id), start=1)
    }

    new_buses = [b.model_copy(update={"bus_id": bus_id_map[b.bus_id]}) for b in buses]
    new_lines = [
        ln.model_copy(update={
            "line_id": line_id_map[ln.line_id],
            "from_bus": bus_id_map[ln.from_bus],
            "to_bus": bus_id_map[ln.to_bus],
        })
        for ln in lines
    ]
    new_transformers = [
        t.model_copy(update={
            "trafo_id": trafo_id_map[t.trafo_id],
            "hv_bus": bus_id_map[t.hv_bus],
            "lv_bus": bus_id_map[t.lv_bus],
        })
        for t in transformers
    ]
    new_household_bus_ids = [bus_id_map[b] for b in household_bus_ids]
    new_household_load_profile_kw = {bus_id_map[b]: v for b, v in household_load_profile_kw.items()}
    new_ev_availability = {bus_id_map[b]: v for b, v in ev_availability.items()}
    new_household_devices = {
        bus_id_map[b]: dev.model_copy(update={"bus_id": bus_id_map[b]})
        for b, dev in household_devices.items()
    }

    return (new_buses, new_lines, new_transformers, new_household_bus_ids,
            new_household_load_profile_kw, new_ev_availability, new_household_devices)


class StubNetworkBuilder(NetworkBuilderProtocol):
    """
    Loads a GridNetwork from a JSON file. Defaults to the small test stub
    (settings.stub_network_path); pass `path` to load a larger feeder
    (e.g. data/feeder_20.json) or a map_ui-exported OSM network.
    """

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path is not None else settings.stub_network_path

    def build(self) -> GridNetwork:
        with open(self._path) as f:
            return GridNetwork.model_validate(json.load(f))


class FixedNetworkBuilder(NetworkBuilderProtocol):
    """
    Wraps an already-built GridNetwork as an injectable builder — for callers
    that already have a GridNetwork (loaded from JSON, extracted via
    OSMNetworkBuilder.build_single_feeder, etc.) and just need to satisfy
    GridEnv's `builder=...` slot without re-running GridCreator or re-reading
    a file on every construction.
    """

    def __init__(self, network: GridNetwork):
        self._network = network

    def build(self) -> GridNetwork:
        return self._network


class OSMNetworkBuilder(NetworkBuilderProtocol):
    """
    Builds a GridNetwork for a bounding box using the vendored GridCreator
    tool (vendor/GridCreator, https://github.com/INATECH-CIG/GridCreator).

    GridCreator depends on a separate stack (pypsa, geopandas, osmnx,
    pycity_base, ...) that is installed in its own conda env, not in
    GridKIT's own venv. It is therefore always run out-of-process via
    `conda run`, never imported directly — vendor/GridCreator/main.py also
    isn't import-safe (it has plotting code at module level, outside any
    `if __name__ == "__main__"` guard).
    """

    def __init__(
        self,
        top: float,
        bottom: float,
        left: float,
        right: float,
        scenario: str,
        technologies: list[str] | None = None,
        load_method: int = 0,
        gridcreator_dir: Path | None = None,
        conda_env: str = "GridCreator",
    ):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.scenario = scenario
        self.technologies = technologies or ["solar", "E_car", "HP"]
        self.load_method = load_method
        self.gridcreator_dir = gridcreator_dir or (
            Path(__file__).resolve().parents[3] / "vendor" / "GridCreator"
        )
        self.conda_env = conda_env

    def build(self) -> GridNetwork:
        self._run_gridcreator()
        grid, buses_df = self._load_output()
        return self._to_grid_network(grid, buses_df)

    def build_single_feeder(self, max_households: int | None = None, network: GridNetwork | None = None) -> GridNetwork:
        """
        Build the full (possibly multi-transformer) GridCreator network, then
        extract just one transformer's radial feeder. Pass an already-built
        `network` (e.g. from a prior build() call) to avoid re-running
        GridCreator just to pick a feeder from the same network.

        RadialPowerFlow (grid_model.surrogate) and build_pypsa_network's slack
        generator both only ever look at transformers[0] — they assume a
        single-transformer network. ding0 LV feeders under different
        transformers are physically disjoint radial trees (no lines between
        them), so rather than silently mis-modelling the other feeders, this
        picks exactly one and returns just its subnet.

        Picks the feeder with the most households at or below
        max_households; if none qualify, falls back to the smallest feeder
        overall. With max_households=None, picks the largest feeder.
        """
        network = network or self.build()
        counts = self.feeder_household_counts(network)

        if max_households is None:
            chosen_trafo = max(counts, key=counts.get)
        else:
            eligible = {t: n for t, n in counts.items() if n <= max_households}
            chosen_trafo = max(eligible, key=eligible.get) if eligible else min(counts, key=counts.get)

        return self.extract_feeder(network, chosen_trafo)

    def list_feeders(self, network: GridNetwork) -> list[FeederSummary]:
        """
        One FeederSummary per transformer — bus membership included so a UI
        can highlight each feeder (e.g. via BusModel.x_coord/y_coord on the
        listed bus_ids) without re-deriving the traversal itself.
        """
        household_ids = set(network.household_bus_ids)
        summaries = []
        for trafo in network.transformers:
            bus_ids, _ = self._feeder_traversal(network, trafo)
            summaries.append(FeederSummary(
                trafo_id=trafo.trafo_id,
                household_count=len(bus_ids & household_ids),
                bus_ids=sorted(bus_ids),
            ))
        return summaries

    def feeder_household_counts(self, network: GridNetwork) -> dict[str, int]:
        """Number of household buses reachable from each transformer's LV bus, following only Lines."""
        return {f.trafo_id: f.household_count for f in self.list_feeders(network)}

    def extract_feeder(self, network: GridNetwork, trafo_id: str) -> GridNetwork:
        """Return a GridNetwork containing only the single-transformer radial feeder for trafo_id."""
        trafo = next(t for t in network.transformers if t.trafo_id == trafo_id)
        bus_ids, line_ids = self._feeder_traversal(network, trafo)

        return GridNetwork(
            network_id=f"{network.network_id}_{trafo_id}",
            buses=[b for b in network.buses if b.bus_id in bus_ids],
            lines=[ln for ln in network.lines if ln.line_id in line_ids],
            transformers=[trafo],
            area_name=network.area_name,
            household_bus_ids=[b for b in network.household_bus_ids if b in bus_ids],
            household_load_profile_kw={
                b: p for b, p in network.household_load_profile_kw.items() if b in bus_ids
            },
            ev_availability={
                b: a for b, a in network.ev_availability.items() if b in bus_ids
            },
            household_devices={
                b: d for b, d in network.household_devices.items() if b in bus_ids
            },
        )

    @staticmethod
    def _feeder_traversal(network: GridNetwork, trafo: TransformerModel) -> tuple[set[str], set[str]]:
        """BFS from a transformer's LV bus along Lines. Returns (bus_ids, line_ids) reachable."""
        adjacency: dict[str, list[tuple[str, str]]] = {b.bus_id: [] for b in network.buses}
        for ln in network.lines:
            adjacency[ln.from_bus].append((ln.to_bus, ln.line_id))
            adjacency[ln.to_bus].append((ln.from_bus, ln.line_id))

        reached = {trafo.lv_bus, trafo.hv_bus}
        queue = [trafo.lv_bus]
        line_ids: set[str] = set()
        while queue:
            bus = queue.pop()
            for nbr, line_id in adjacency[bus]:
                line_ids.add(line_id)
                if nbr not in reached:
                    reached.add(nbr)
                    queue.append(nbr)
        return reached, line_ids

    def _run_gridcreator(self) -> None:
        # Mirrors GridCreator(steps=[1,2,3,4,5]) in vendor/GridCreator/main.py
        # step by step, since main.py itself can't be imported safely.
        driver = f"""# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

proj_data = Path(sys.prefix) / "Library" / "share" / "proj"
if proj_data.exists():
    os.environ["PROJ_DATA"] = str(proj_data)
    os.environ["PROJ_LIB"] = str(proj_data)

    import pyproj
    pyproj.datadir.set_data_dir(str(proj_data))

import main_functions as mf
import input_data as data

data.save_data()
input_path = os.path.join(os.getcwd(), 'input')

bbox = [{self.left}, {self.bottom}, {self.right}, {self.top}]
# Single ding0 pass, NOT mf.ding0_grid(): that re-runs load_grid over the
# extent of the first result. Once that extent touches a neighbouring MV grid,
# load_buses_in_bbox keeps only the LAST folder with a hit in os.listdir order
# (filesystem-dependent: alphabetical on Windows, arbitrary on macOS/Linux) and
# silently swaps in a network that was never inside the user's box. The first
# pass already pulls in each touched feeder whole, so the second adds nothing.
grid = mf.ding0.load_grid(bbox, os.path.join(input_path, 'grids'))
bbox = mf.func.compute_bbox_from_buses(grid)
grid.name = {self.scenario!r}

buses_df, area, features = mf.osm_data(grid, bbox, 0.0002)
buses_df = mf.data_assignment(buses_df, input_path)

gcp = {self.technologies!r}
buses_df, factor_bbox = mf.gcp_assignment(buses_df, gcp, input_path)
buses_df = mf.appartments_assignment(buses_df)
buses_df = mf.gcp_fill(buses_df, gcp, factor_bbox, input_path)

grid = mf.loads_assignment(grid, buses_df, bbox, input_path, {self.load_method!r})
grid = mf.pypsa_preparation(grid)

# Only write what OSMNetworkBuilder._load_output() actually reads (the pypsa
# CSV export and buses_df) rather than calling ding0_grid_generator.save_output_data(),
# which also writes area.gpkg/features.gpkg — geopandas/pyogrio exports of raw
# OSM tag data we never use, and which can fail on arbitrary OSM tag values
# unrelated to anything we need.
output_dir = os.path.join('output', {self.scenario!r}, 'step_5')
os.makedirs(output_dir, exist_ok=True)
if not grid.buses.empty:
    grid.export_to_csv_folder(os.path.join(output_dir, 'grid'))
buses_df.to_csv(os.path.join(output_dir, 'buses.csv'))
"""
        # `conda run -n ENV python -c "<multiline>"` fails on Windows with
        # NotImplementedError: Support for scripts where arguments contain
        # newlines not implemented — conda's Windows subprocess handling
        # can't pass a multi-line string as a single -c argument. Writing
        # the driver to a real .py file next to GridCreator and running
        # that instead works on every platform.
        fd, driver_path_str = tempfile.mkstemp(
            suffix=".py", prefix="_gridkit_gridcreator_driver_", dir=self.gridcreator_dir
        )
        driver_path = Path(driver_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(driver)
            env_name = (self.conda_env or "GridCreator").strip()

            gridcreator_python_env = os.environ.get("GRIDCREATOR_PYTHON")

            if gridcreator_python_env:
                cmd = [gridcreator_python_env, driver_path.name]
            else:
                cmd = ["conda", "run", "-n", env_name, "python", driver_path.name]

            run_env = os.environ.copy()
            result = subprocess.run(
                cmd,
                cwd=self.gridcreator_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=run_env,
            )
        finally:
            driver_path.unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"GridCreator run failed (exit {result.returncode}):\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )

    def _load_output(self) -> tuple[pypsa.Network, pd.DataFrame]:
        step_dir = self.gridcreator_dir / "output" / self.scenario / "step_5"
        grid = pypsa.Network()
        grid.import_from_csv_folder(step_dir / "grid")
        buses_df = pd.read_csv(step_dir / "buses.csv", index_col=0)
        return grid, buses_df

    def _to_grid_network(self, grid: pypsa.Network, buses_df: pd.DataFrame) -> GridNetwork:
        buses = [
            BusModel(
                bus_id=str(name),
                v_nom_kv=row["v_nom"],
                x_coord=None if pd.isna(row["x"]) else float(row["x"]),
                y_coord=None if pd.isna(row["y"]) else float(row["y"]),
            )
            for name, row in grid.buses.iterrows()
        ]

        lines = []
        for name, row in grid.lines.iterrows():
            # pypsa gives absolute r/x in Ohm; GridNetwork wants per-km values.
            # A handful of ding0 jumper lines have length == 0 — fall back to
            # a short nominal length rather than dividing by zero.
            length_km = row["length"] or 0.001
            v_nom_kv = grid.buses.at[row["bus0"], "v_nom"]
            max_i_ka = row["s_nom"] / (math.sqrt(3) * v_nom_kv) if v_nom_kv else 0.0
            lines.append(
                LineModel(
                    line_id=str(name),
                    from_bus=str(row["bus0"]),
                    to_bus=str(row["bus1"]),
                    length_km=length_km,
                    r_ohm_per_km=row["r"] / length_km,
                    x_ohm_per_km=row["x"] / length_km,
                    max_i_ka=max_i_ka,
                )
            )

        transformers = [
            TransformerModel(
                trafo_id=str(name),
                hv_bus=str(row["bus0"]),
                lv_bus=str(row["bus1"]),
                s_nom_mva=row["s_nom"],
                vn_hv_kv=grid.buses.at[row["bus0"], "v_nom"],
                vn_lv_kv=grid.buses.at[row["bus1"], "v_nom"],
            )
            for name, row in grid.transformers.iterrows()
        ]

        household_bus_ids = (
            buses_df.index[buses_df["Haushalte"] > 0].astype(str).tolist()
        )

        has_profile = set(grid.loads_t.p_set.columns)
        household_load_profile_kw: dict[str, list[float]] = {}
        for bus in household_bus_ids:
            # A bus can host several households (several "{bus}_load_N"
            # entries) — sum them into one household-bus-level demand curve.
            load_names = [
                name for name, row in grid.loads.iterrows()
                if row["bus"] == bus and name in has_profile
            ]
            if load_names:
                total_mw = grid.loads_t.p_set[load_names].sum(axis=1)
                household_load_profile_kw[bus] = _first_episode_day(total_mw * 1000.0)

        ev_availability: dict[str, list[bool]] = {}
        for bus in household_bus_ids:
            charge_link = f"{bus}{EV_BUS_SUFFIX}_Connector_charge"
            if charge_link in grid.links_t.p_max_pu.columns:
                availability = _first_episode_day(grid.links_t.p_max_pu[charge_link])
                ev_availability[bus] = [bool(round(v)) for v in availability]

        # Real per-household device assignment from GridCreator's own
        # gcp_assignment/gcp_fill step (see _run_gridcreator's driver
        # script above) — Power_solar/storage/Power_E_car/Power_HP are
        # written into buses_df for every bus, 0 meaning "none".
        buses_df_by_str_id = buses_df.set_axis(buses_df.index.astype(str))
        household_devices: dict[str, HouseholdDevices] = {}
        for bus in household_bus_ids:
            row = buses_df_by_str_id.loc[bus]
            pv_kw = float(row.get("Power_solar", 0.0) or 0.0)
            storage_kwh = float(row.get("storage", 0.0) or 0.0)
            ev_kw = float(row.get("Power_E_car", 0.0) or 0.0)
            hp_kw = float(row.get("Power_HP", 0.0) or 0.0)
            household_devices[bus] = HouseholdDevices(
                bus_id=bus,
                ev=ev_kw > 0,
                heat_pump=hp_kw > 0,
                pv=pv_kw > 0,
                pv_kwp=(
                    _clamp(pv_kw, const.PV_PEAK_KWP_MIN, const.PV_PEAK_KWP_MAX) if pv_kw > 0 else None
                ),
                battery=storage_kwh > 0,
                battery_kwh=(
                    _clamp(storage_kwh, const.BATTERY_CAPACITY_KWH_MIN, const.BATTERY_CAPACITY_KWH_MAX)
                    if storage_kwh > 0 else None
                ),
            )

        (buses, lines, transformers, household_bus_ids,
         household_load_profile_kw, ev_availability, household_devices) = assign_clean_ids(
            buses, lines, transformers, household_bus_ids,
            household_load_profile_kw, ev_availability, household_devices,
        )

        return GridNetwork(
            network_id=self.scenario,
            buses=buses,
            lines=lines,
            transformers=transformers,
            area_name=self.scenario,
            household_bus_ids=household_bus_ids,
            household_load_profile_kw=household_load_profile_kw,
            ev_availability=ev_availability,
            household_devices=household_devices,
        )
