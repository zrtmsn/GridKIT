# GridKIT

GridKIT simulates a low-voltage (0.4 kV) household grid and trains multi-agent
reinforcement learning (IPPO) to manage flexible devices (EVs, heat pumps)
under §14a EnWG curtailment rules — the grid operator's right to temporarily
dim a household's draw when the local transformer/line would overload.

This guide takes you from a fresh clone to: building a network in the map UI,
training on it, and viewing the results — all commands verified on this
branch, not copied from an older/aspirational doc.

---

## 1. Setup

**Prerequisites**: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), git.
[conda](https://docs.conda.io) is only needed if you'll build networks from
real OpenStreetMap data (see §3) — skip it if you're just running the stub
network or training on an already-downloaded `grid_network.json`.

```bash
git clone <repo-url>
cd GridKIT
git submodule update --init --recursive   # fetches vendor/GridCreator

uv sync                          # creates .venv, installs everything from pyproject.toml
source .venv/bin/activate        # .venv\Scripts\activate on Windows
```

### The one thing to know about running any command here

Two import styles coexist in this codebase: `core/`, `grid_model/`,
`map_ui/`, `scenarios/` import each other bare (`from core.models import
...`), which resolves with `src/GridKIT` on `PYTHONPATH`; `rl_engine/` and
`scripts/` import with the full package prefix (`from GridKIT.core import
...`), which resolves with `src` on `PYTHONPATH`. Every command below sets
**both**:

```bash
export PYTHONPATH=src/GridKIT:src
```

Set it once per terminal session (or prefix each command with it) and every
command in this guide works as written. `pytest` doesn't need this — it's
already configured in `pyproject.toml`.

### Verify the setup

```bash
pytest                                                          # full test suite
python -m grid_model.cli build --stub --out /tmp/net.json       # builds the bundled 5-household stub network
python -m grid_model.cli plot /tmp/net.json --out-dir /tmp/demo # runs a quick heuristic episode, saves PNGs
```

If both of those work, your environment is set up correctly — everything
past this point either needs conda (map UI, real networks) or just takes
longer (training).

---

## 2. Map UI — build a network

```bash
streamlit run src/GridKIT/map_ui/map_widget.py
```

Opens at `http://localhost:8501`. In the sidebar: search a place or draw a
box on the map (or type a bounding box manually) → **"GridNetwork
erzeugen"**. Once built, the page shows the network's stats, a topology
visualization, and a household-configuration section (EV/heat-pump
penetration sliders + per-household overrides) with **download buttons** for
`grid_network.json` and `household_configuration.json` — save those
somewhere, you'll need `grid_network.json` for training in §3.

### This needs GridCreator, which needs conda

Building from a real drawn area (not the stub) shells out to the vendored
GridCreator tool via `conda run -n GridCreator ...`. One-time setup:

```bash
conda create -n GridCreator python=3.12.11
conda activate GridCreator
cd vendor/GridCreator
pip install -r requirements.txt
cd ../..
```

GridCreator also needs its input data present: download `input.zip` from
https://zenodo.org/records/17884917 and unpack it so you end up with
`vendor/GridCreator/input/` containing the `grids`/`weather_2013`/
`zensus_daten` subfolders.

Don't want to set up conda? Point at any Python interpreter that already has
GridCreator's `requirements.txt` installed instead, and it'll be called
directly (no conda involved):

```bash
GRIDCREATOR_PYTHON=/path/to/that/python streamlit run src/GridKIT/map_ui/map_widget.py
```

**Known issue**: a fresh `vendor/GridCreator` checkout can fail with
`TypeError: Invalid value '...' for dtype 'int64'` inside
`appartments_assignment` (a pandas-version incompatibility in
`main_functions.py` — an int column initialized as `0` later receives
fractional writes). Fix: in `vendor/GridCreator`, change
`buses['Bewohnerinnen'] = 0` to `buses['Bewohnerinnen'] = 0.0`.

---

## 3. Training

```bash
python -m GridKIT.scripts.run_experiment --network data/stub_network.json --out outputs
```

Trains a separate IPPO policy for each EV-penetration level in
`core.constants.EV_PENETRATION_LEVELS` (20/40/60% by default) on that
network, evaluates it against two rule-based baselines (flat/immediate
charging, price-following), and writes `outputs/summary.json`,
`outputs/timelines.json`, and a trained checkpoint per penetration level
under `outputs/checkpoints/`.

Defaults are `--iterations 40 --seeds 12` per penetration level — a real
training run, expect it to take a while. For a quick check that everything
wires up correctly, cut both down:

```bash
python -m GridKIT.scripts.run_experiment --network data/stub_network.json --out outputs --iterations 5 --seeds 2
```

No `--network`? It defaults to the bundled 20-household `data/feeder_20.json`
— useful for a first run without needing a real GridCreator build at all.

**Caveat**: this script drives device penetration itself (the `--iterations`
run above assigns EV/battery/heat-pump/PV to the same evenly-spread
households per penetration level) — it does **not** read the
`household_configuration.json` you can download from the map UI's
per-household editor. The two aren't wired together yet.

---

## 4. Viewing results

```bash
python -m GridKIT.scripts.plot_results --results outputs --out outputs/graphs
```

Reads `outputs/summary.json`/`outputs/timelines.json` from §3 and saves PNGs
to `outputs/graphs/` — curtailment/SoC/reward comparisons across scenarios
and penetration levels, device-power timelines. No Streamlit/dashboard
needed; this is a plain headless script.

If you just want a fast sanity-check plot for a network **without** training
an RL policy at all (a simple heuristic charging policy instead), use the CLI
from §1 again:

```bash
python -m grid_model.cli plot data/stub_network.json --out-dir outputs/quicklook
```

This prints a one-line episode summary (curtailment events, peak transformer
loading, EV targets met) and saves 4 PNGs (network topology, loading,
load+curtailment, device SoC) in seconds — good for confirming a newly built
network is sane before committing to a full training run.

---

## 5. Dashboard

Interactive web UI for exploring training results:

```bash
streamlit run src/GridKIT/dashboard_prototype/app.py
```

Opens at `http://localhost:8502`. The dashboard provides:

| Tab | Description |
|-----|-------------|
| 🗺️ Overload Map | Interactive Folium map showing overloaded lines/transformers per timestep |
| 📊 Metrics | Static PNG charts from `plot_results` (curtailment, SoC, rewards, device power) |
| 🎯 Training | Interactive Plotly charts showing RLlib training metrics (rewards, loss, entropy) |

**Requirements:** Training must be completed first (§3) to generate `summary.json`, `timelines.json`, and `iteration_metrics.json`.

---

## Project structure

```
src/GridKIT/
├── core/                 # shared pydantic models, settings, constants, Protocol interfaces
├── grid_model/           # network construction (OSM/GridCreator/stub) + episode simulation
├── map_ui/               # Streamlit + Folium: build a network, configure households
├── rl_engine/            # RLlib IPPO training against core.protocols.GridEnvProtocol
├── scenarios/            # rule-based baseline policies + the episode/scenario runner
├── dashboard_prototype/  # Streamlit dashboard: explore training results interactively
└── scripts/              # entry points: run_experiment (train), plot_results (headless plots)
```

Modules besides `core` don't import each other's internals — `rl_engine` and
`scenarios` are written against `core.protocols.GridEnvProtocol` and receive
a concrete environment (`grid_model.GridEnv`) only by dependency injection.

## Development

```bash
pytest                                              # full suite
pytest src/GridKIT/grid_model/test_environment.py   # one file
pytest -k battery                                   # by keyword
```

Tests live next to the module they cover (`grid_model/test_environment.py`),
not in a separate `tests/` tree. There is no configured linter/formatter.

---

## Libraries

| Library | Used in | Purpose |
|---|---|---|
| [GridCreator](https://github.com/INATECHCIG/GridCreator) | `grid_model` | Real ding0 LV grid topology for a drawn area |
| [pandapower](https://pandapower.readthedocs.io) | `grid_model` | Power flow simulation |
| [Gymnasium](https://gymnasium.farama.org) | `rl_engine` | RL environment interface |
| [RLlib](https://docs.ray.io/en/latest/rllib/) or [SB3](https://stable-baselines3.readthedocs.io) | `rl_engine` | MARL agents |
| [Streamlit](https://streamlit.io) | `dashboard` | UI and visualization |

## Enabling real ding0 grids (recommended)

When you draw an area in the grid designer, GridKIT extracts the **actual** LV grid
for that box from the ding0 archive — every transformer real and individually sized
(GridCreator step 1). Without the archive it falls back to the OSM builder, which can
only place **one generic 160 kVA transformer** for the whole area, so its congestion
and §14a curtailment numbers are not physically meaningful.

To enable it, download `input.zip` from [Zenodo](https://zenodo.org/records/17884917)
and unpack it so the grids land here:

```
vendor/GridCreator/input/grids/<grid_district>/topology/buses.csv
```

Or point `$GRIDKIT_DING0_GRIDS_DIR` at an existing copy. The designer detects the
archive automatically and greys out the ding0 option when it is missing.
Coverage is Germany-only, limited to the districts in your download.
