# GridKIT

GridKIT simulates a low-voltage (0.4 kV) household grid and trains multi-agent
reinforcement learning (IPPO) to manage flexible devices — EVs, home
batteries, heat pumps — under §14a EnWG curtailment rules. A fast linear
power-flow surrogate drives the per-step grid physics; real network
topologies are built from OpenStreetMap data via the vendored
[GridCreator](https://github.com/INATECH-CIG/GridCreator) tool.

For a full architecture breakdown (what each module does, which data
contracts are shared, which paths are actually wired up vs. stubs), see
[`CLAUDE.md`](CLAUDE.md) — it's kept up to date with the actual code, not
aspirational.

---

## Setup

You need **two** separate environments:

- a **uv-managed venv** for GridKIT itself (Python 3.12+)
- a **conda env** for GridCreator (only needed if you want to build real
  networks from OpenStreetMap — GridKIT calls it out-of-process via
  `conda run -n GridCreator`, so it never has to be activated by hand, just
  present on the machine)

Skip the conda/GridCreator steps if you only want to run the test suite or
work with the bundled stub networks (`data/stub_network.json`,
`data/feeder_20.json`).

### 1. Clone the repo and its submodule

```bash
git clone <repo-url>
cd GridKIT
git submodule update --init --recursive   # fetches vendor/GridCreator
```

### 2. Install uv

If you don't already have [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   (Windows)
```

### 3. Create and activate the GridKIT venv

```bash
uv sync                        # creates .venv/, installs pyproject.toml deps
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

Verify with the test suite:

```bash
pytest
```

### 4. Install conda (only needed for real OSM/ding0 grid builds)

If you don't already have conda, install
[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).

### 5. Create the GridCreator conda env

GridCreator has its own dependency stack and is never imported into
GridKIT's own venv, so it gets its own environment:

```bash
conda create -n GridCreator python=3.12.11
conda activate GridCreator
pip install -r vendor/GridCreator/requirements.txt
conda deactivate
```

GridCreator also needs its input data (ding0 grids, Zensus, weather) — see
`vendor/GridCreator/README.md`'s "Necessary input data" section for the
download link and where to unpack it (`vendor/GridCreator/input/`).

---

## Running it

With the GridKIT venv (`.venv`) active, from the repo root:

```bash
streamlit run src/GridKIT/scripts/app.py
```

This is the main entry point: one Streamlit app with two pages —

- **Karte** — search/draw a bounding box, build a `GridNetwork` from real OSM
  data (via the GridCreator conda env), configure households (EV/heat pump),
  and optionally save it and kick off IPPO training in the background.
- **Dashboard** — browse every saved run, watch training progress live, and
  view results (curtailment, SoC satisfaction, transformer peak loading,
  reward, bill) once a run finishes.

Saved networks/runs live under `runs/<run_id>/` (gitignored).

### Other useful commands

```bash
pytest                                  # full test suite
pytest -k battery                       # by keyword
pytest src/GridKIT/grid_model/test_environment.py   # one file

# grid_model CLI (needs both roots on PYTHONPATH outside pytest)
PYTHONPATH=src/GridKIT:src python -m grid_model.cli build --stub --out /tmp/net.json
PYTHONPATH=src/GridKIT:src python -m grid_model.cli plot /tmp/net.json --out-dir outputs/demo

# map UI standalone (no dashboard page)
streamlit run src/GridKIT/map_ui/map_widget.py
```

See `grid_model/cli.py`'s module docstring for more CLI examples (OSM,
ding0, add-bus, add-household, add-ev), and `CLAUDE.md` for the full module
map and the two coexisting import styles (`from core...` vs.
`from GridKIT.core...`) you'll hit if you run a script directly instead of
through `pytest` or `streamlit run`.

---

## Project structure

```
src/GridKIT/
├── core/          # shared pydantic models, settings, constants, Protocol interfaces
├── grid_model/    # network construction + episode simulation (the only module that owns grid physics)
├── map_ui/        # Streamlit + Folium area picker → GridNetwork
├── rl_engine/     # RLlib IPPO training against GridEnvProtocol
├── scenarios/     # rule-based baseline policies + episode runner
├── dashboard/     # Streamlit viewer for saved runs
└── scripts/       # entry points (app.py, train_run.py, run_experiment.py, ...)
```

Modules other than `core` don't import each other's internals — `rl_engine`
and `scenarios` are written against `core.protocols.GridEnvProtocol` and
receive a concrete env by dependency injection, never by importing
`grid_model` directly. See `CLAUDE.md` for the reasoning and for which parts
of `rl_engine`/`scripts` are fully wired up vs. still stubs (e.g. DQN is
configured but never implemented; only IPPO is real).
