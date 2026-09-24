
# GridKIT

GridKIT simulates a low-voltage (0.4 kV) household grid and trains multi-agent
reinforcement learning (IPPO) to manage flexible devices — EVs, home
batteries, heat pumps — under §14a EnWG curtailment rules. A fast linear
power-flow surrogate drives the per-step grid physics; real network
topologies are built from OpenStreetMap data via the vendored
[GridCreator](https://github.com/INATECH-CIG/GridCreator) tool.

---

## Setup Requirements

- ~40 GB free disk space
- Internet connection
- Windows, Linux or MacOS as operating system

## Automated setup (Windows x86-64)

On Windows everything above is automated by `setup_windows_GER.ps1` in the
repo root (German console output, matching GridKIT's German UI):

```powershell
# after installing the VC++ Redistributable (step 0) once, in PowerShell:
powershell -ExecutionPolicy ByPass -File setup_windows_GER.ps1
```

The script:

1. checks for the VC++ runtime and aborts with instructions if missing,
2. initializes the `vendor/GridCreator` submodule,
3. installs `uv` if absent and runs `uv sync`,
4. runs the test suite (`pytest`),
5. installs Miniconda per-user (no admin rights) and puts it on your `PATH`,
6. creates the `GridCreator` conda env (`python=3.12.11` + requirements),
7. downloads the GridCreator input data from Zenodo (~2.9 GB) with MD5
   verification and unpacks it (~18 GB) — skip with `-SkipInputData` and
   re-run later without the switch.

After the script finishes: **open a new shell**, `cd` into the repo root,
activate the venv (`.venv\Scripts\activate` in cmd) and start Streamlit as
above. Linux and macOS users follow the manual steps — no automated script
is provided for those platforms (smaller user base).

Should the script fail during a specific step, you can use the manual setup as a fallback, to continue from that step onward.

## Manual setup (Windows / Linux / MacOS)

### 0. WINDOWS ONLY pre-installation requirement: VC++ Redistributable

On Windows, imports of `torch` & Co. fail without the
**VC++ 2015–2022 Redistributable (x64)**. Install it once before anything
else (requires admin rights):

1. Download: <https://aka.ms/vs/17/release/vc_redist.x64.exe>
   (official overview: [latest-supported-vc-redist](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170))
2. Run the file and confirm the UAC prompt.

No equivalent step is needed on Linux/macOS.

### 1. Clone the repo and its submodule

```bash
git clone https://github.com/zrtmsn/GridKIT
cd GridKIT # navigate to GridKIT's root. All following commands shall be executed from here, unless remarked otherwise.
git submodule update --init --recursive   # fetches vendor/GridCreator
```

### 2. Install uv

If you don't already have [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                          # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

> **Windows:** start a **new shell** after installing so that `uv` is found
> on your `PATH`.

### 3. Create and activate the GridKIT venv

```bash
uv sync                        # creates .venv/, installs pyproject.toml dependencies

source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows (cmd)
```

Verify with the test suite:

```bash
pytest
```

> If `pytest` fails on Windows with DLL/import errors, go back to
> [step 0](#0-windows-only-vc-redistributable-do-this-first) — a missing
> VC++ runtime is the usual cause.

### 4. Install conda
*This step does not require you to operate in GridKIT’s root directory*

If you don't already have conda, install
[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) by following their guides.

- **Windows:** the per-user ("JustMe") installer needs no admin rights.
  You do **not** need the special "Anaconda Prompt" — GridKIT calls conda
  out-of-process, it just has to be on your `PATH` (the installer's
  "add to PATH" option, or the automated setup script, handles this).
	- Alternatively: **Start a conda shell from Windows Start Menu** – it should be added to your Start Menu after install automatically. If you do not want to use the "add to PATH" option, always make sure to run GridKIT from a conda shell via the Windows Start Menu.
- **Linux:** installing conda via your distribution's package manager is
  the most reliable route; a standalone Miniconda install may not be picked
  up correctly if the PATH is falsely entered to .\bashrc (only recommended for experienced users)

### 5. Create the GridCreator conda env
*From here: Again required to operate in GridKIT's root directory*

GridCreator has its own dependency stack and is never imported into
GridKIT's own venv, so it gets its own environment:

```bash
conda create -y -n GridCreator python=3.12.11
conda activate GridCreator
pip install -r vendor/GridCreator/requirements.txt
conda deactivate
```

*More information for advanced users:*

> **Note:** `-y` silently **overwrites** an existing env named `GridCreator`.
> Drop the flag if you want to be asked first.

> **Note (conda ≥ 25):** conda may interactively ask you to accept the terms
> of service. For non-interactive use set
> `CONDA_PLUGINS_AUTO_ACCEPT_TOS=true` first.

### 6. GridCreator input data (~2.9 GB download, ~18 GB extracted)

GridCreator needs its input data (ding0 grids, Zensus, weather). Download
`input.zip` from Zenodo:

- <https://zenodo.org/records/17884917/files/input.zip?download=1>

Unpack it into `vendor/GridCreator/` — the zip contains `input/` at its
root, so afterwards you should have:

```
vendor/GridCreator/input/grids
vendor/GridCreator/input/weather_2013
vendor/GridCreator/input/zensus_daten
```

> [!IMPORTANT]
> Please make sure to have all zipped files in the `vendor/GridCreator/input/` directory. Otherwise, you may not be able to create GridNetworks and will likely run into errors.

---

## Running GridKIT


> [!IMPORTANT]
> No matter which setup you chose, make sure to open a new shell and continue from there. This is required to let `PATH` changes take full effect and will likely lead to conflicts with running Conda if skipped.


**Windows**
```bash
cd GridKIT									# navigate to GridKIT's root directory
.venv\Scripts\activate					    # activate venv
streamlit run src/GridKIT/scripts/app.py	# run web app
```

**Linux / MacOS**
```bash
cd GridKIT									# navigate to GridKIT's root directory
source .venv/bin/activate					# activate venv
streamlit run src/GridKIT/scripts/app.py	# run web app
```


**Then:** After executing the Streamlit command, press Enter – Streamlit will then show you a link to the locally hosted Web UI that serves as GridKIT's main entry point. 
(For instance: http://localhost:8501)


A quick overview of the two tabs found in the Streamlit app:

- **Karte**: search/draw a bounding box, build a `GridNetwork` from real OSM
  data (via the GridCreator conda env), configure households (EV/heat pump),
  and optionally save it and kick off IPPO training in the background.
- **Dashboard**: browse every saved run, watch training progress live, and
  once a run finishes explore its results across five tabs (German UI):
	- **Überblick** (does the grid hold, and which scenario is best),
	- **Netzauslastung** (loading over the day, duration curve, which element
	  overloads when),
	- **Geräte & Haushalte** (per-device power, SoC, heat-pump
	  comfort, a representative household),
	- **Szenarienvergleich** (the strategies
	  side by side plus the overload map)
	- **Training** (reward convergence,
	  reward against the baselines, policy entropy). Every chart exports to
	  CSV/JSON.

Saved networks/runs live under the directories `runs/<run_id>/` (gitignored).

### For Testing via CLI: A Batch experiment (no map)

For testing the training, GridKIT provides a sweep on scenarios across 
three fixed percentages of EV-Households present on a fixed network,
independent of the map UI:

```bash
# From the repo root, with the venv activated, expose the source roots first
# (GridKIT is not pip-installed, so `GridKIT` is not on sys.path by default):
export PYTHONPATH="$(pwd)/src/GridKIT:$(pwd)/src"
python -m GridKIT.scripts.run_experiment          # writes outputs/ 
streamlit run src/GridKIT/dashboard/app.py        # same five tabs, on outputs/
```
**NOTE:** For a quick test run with max of 3 iterations, use param `--max-iterations 3`
as in `python -m GridKIT.scripts.run_experiment --max-iterations 3`

**NOTE:** `run_experiment` only **records** overload information (`overloaded_lines` /
`line_overload_steps` in `summary.json` / `timelines.json`) — it does not check it.
To verify that every curtailment step coincides with an overload, run the consistency
check on the results directory you passed to `--out` (default `outputs`; the check
script itself defaults to `outputs_test_weak`):

```bash
python src/GridKIT/scripts/test_overload_logging.py outputs
```

`GRIDKIT_OUTPUT_DIR` points the dashboard at a different results directory.
When both `runs/` and `outputs/` hold results, the standalone dashboard offers
a source picker in the sidebar.

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

---

## Project structure

```
src/GridKIT/
├── core/          # shared pydantic models, settings, constants, Protocol interfaces
├── grid_model/    # network construction + episode simulation (the only module that owns grid physics)
├── map_ui/        # Streamlit + Folium area picker → GridNetwork
├── rl_engine/     # RLlib IPPO training
├── scenarios/     # rule-based baseline policies + episode runner
├── dashboard/     # Streamlit viewer for saved runs
└── scripts/       # entry points (app.py, train_run.py, run_experiment.py, ...)
```


## Acknowledgements

GridKIT builds on several open-source projects; their licences govern our
own distribution, so credit goes to:

| Project                                                                        | Used for                                      | License                                         |
| ------------------------------------------------------------------------------ | --------------------------------------------- | ----------------------------------------------- |
| [GridCreator](https://github.com/INATECH-CIG/GridCreator) (vendored submodule) | Real OSM/ding0 low-voltage network generation | GPL v3.0                                        |
| [RLlib](https://docs.ray.io/en/latest/rllib/) (Ray)                            | Multi-agent IPPO training engine              | Apache-2.0                                      |
| [uv](https://docs.astral.sh/uv/)                                               | Python environment & dependency management    | MIT                                             |
| [Streamlit](https://streamlit.io/)                                             | Dashboard & map web UI                        | Apache-2.0                                      |
| [Folium](https://python-visualization.github.io/folium/)                       | Interactive map tiles in the map UI           | MIT                                             |
| [OpenStreetMap](https://www.openstreetmap.org)                                 | Source of real street/network data            | [ODbL](https://www.openstreetmap.org/copyright) |
| [bwUniCluster 3.0 (bwHPC)](https://wiki.bwhpc.de/e/BwUniCluster3.0)            | HPC compute resources for RL training runs    | funded by the state of Baden-Württemberg        |


Regarding GridKIT's licensing, view the repo's `LICENSE` file. 
