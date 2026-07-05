#grid_model/builder.py
import json
import math
import subprocess
from pathlib import Path

import pandas as pd
import pypsa

import core.constants as const
from core import BusModel, GridNetwork, LineModel, TransformerModel, settings
from core.protocols import NetworkBuilderProtocol

EV_BUS_SUFFIX = "_E_Car"


def _resample_to_episode_steps(series: pd.Series) -> list[float]:
    """
    GridCreator's own snapshots are hourly (a full year); GridKIT episodes are
    one representative day at 15-min resolution (EPISODE_STEPS steps). Take
    the first day of the series and forward-fill it onto GridKIT's grid, so
    each source value covers all the finer-grained steps within it.
    """
    source_minutes = (series.index[1] - series.index[0]).total_seconds() / 60.0
    steps_per_day = int(round(24 * 60 / source_minutes))
    day = series.iloc[:steps_per_day]
    target_index = pd.date_range(
        day.index[0], periods=const.EPISODE_STEPS, freq=f"{const.TIMESTEP_MINUTES}min"
    )
    return day.reindex(target_index, method="ffill").tolist()

class StubNetworkBuilder(NetworkBuilderProtocol):

    def build(self) -> GridNetwork:
        """
        loads a json file from data containing example network specs
        and converts it into a GridNetwork as defined in core.models.
        """
        with open(settings.stub_network_path) as f:
            return GridNetwork.model_validate(json.load(f))


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

    def _run_gridcreator(self) -> None:
        # Mirrors GridCreator(steps=[1,2,3,4,5]) in vendor/GridCreator/main.py
        # step by step, since main.py itself can't be imported safely.
        driver = f"""
import os
import main_functions as mf
import ding0_grid_generator
import input_data as data

data.save_data()
input_path = os.path.join(os.getcwd(), 'input')

bbox = [{self.left}, {self.bottom}, {self.right}, {self.top}]
grid, bbox = mf.ding0_grid(bbox, input_path)
grid.name = {self.scenario!r}

buses_df, area, features = mf.osm_data(grid, bbox, 0.0002)
buses_df = mf.data_assignment(buses_df, input_path)

gcp = {self.technologies!r}
buses_df, factor_bbox = mf.gcp_assignment(buses_df, gcp, input_path)
buses_df = mf.appartments_assignment(buses_df)
buses_df = mf.gcp_fill(buses_df, gcp, factor_bbox, input_path)

grid = mf.loads_assignment(grid, buses_df, bbox, input_path, {self.load_method!r})
grid = mf.pypsa_preparation(grid)

ding0_grid_generator.save_output_data(
    grid, buses_df, bbox, area, features,
    scenario={self.scenario!r}, steps=[1, 2, 3, 4, 5], path='output',
)
"""
        result = subprocess.run(
            ["conda", "run", "-n", self.conda_env, "python", "-c", driver],
            cwd=self.gridcreator_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"GridCreator run failed (exit {result.returncode}):\n{result.stderr}"
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
                household_load_profile_kw[bus] = _resample_to_episode_steps(total_mw * 1000.0)

        ev_availability: dict[str, list[bool]] = {}
        for bus in household_bus_ids:
            charge_link = f"{bus}{EV_BUS_SUFFIX}_Connector_charge"
            if charge_link in grid.links_t.p_max_pu.columns:
                availability = _resample_to_episode_steps(grid.links_t.p_max_pu[charge_link])
                ev_availability[bus] = [bool(round(v)) for v in availability]

        return GridNetwork(
            network_id=self.scenario,
            buses=buses,
            lines=lines,
            transformers=transformers,
            area_name=self.scenario,
            household_bus_ids=household_bus_ids,
            household_load_profile_kw=household_load_profile_kw,
            ev_availability=ev_availability,
        )