# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The repo's `README.md` describes an earlier/aspirational version of the project and is out of date — everything below was verified by reading the current code and actually running it (tests, the CLI, and a live RLlib training iteration), not from the README.

## What this is

GridKIT simulates a low-voltage (0.4 kV) household grid and trains multi-agent RL (IPPO) to manage flexible devices under §14a EnWG curtailment rules. Each household can have an EV, a home battery, and a heat pump, each a separate RL agent (device types share one policy each), plus optional exogenous PV. All device-agents at one household share a single reward = minimise net electricity bill − EV-SoC-miss penalty − HP-comfort-miss penalty. A fast linear power-flow surrogate (not a full AC solve) drives the per-step grid physics and triggers proportional curtailment when a transformer/line would overload.

## Commands

```bash
uv sync                                    # install deps + create .venv
source .venv/bin/activate                  # .venv\Scripts\activate on Windows
git submodule update --init --recursive    # fetch vendor/GridCreator (only needed for real OSM/ding0 builds)

pytest                                              # full suite (108 passed, 1 skipped as of this writing, ~25s)
pytest src/GridKIT/grid_model/test_environment.py   # one file
pytest -k battery                                   # by keyword

# grid_model CLI needs both roots on PYTHONPATH (see "two import styles" below)
PYTHONPATH=src/GridKIT:src python -m grid_model.cli build --stub --out /tmp/net.json
PYTHONPATH=src/GridKIT:src python -m grid_model.cli plot /tmp/net.json --out-dir outputs/demo
# full --help / more examples (OSM, ding0, add-bus, add-household, add-ev) are in grid_model/cli.py's module docstring

python -m streamlit run src/GridKIT/map_ui/map_widget.py   # map UI: draw/enter a bbox → build a GridNetwork via GridCreator
```

There is no configured linter/formatter in this repo. Tests live next to the module they cover (`grid_model/test_environment.py`, not a separate `tests/` tree) and are collected by plain `pytest` because `pyproject.toml` sets `pythonpath = ["src/GridKIT", "src"]`.

### Two import styles coexist — both roots must be on `PYTHONPATH` outside pytest

Verified by actually running code, not just reading it: `core/`, `grid_model/`, `map_ui/`, `scenarios/` import each other with bare names (`from core.models import ...`, `from grid_model.builder import ...`), which only resolves with `src/GridKIT` on the path. `rl_engine/`, `scripts/`, and the top-level `training_utils.py`/`test_run_training*.py` instead import with the full package prefix (`from GridKIT.core...`, `from GridKIT.rl_engine...`), which only resolves with `src` on the path. Running a script directly (not via `pytest`, which sets both) needs `PYTHONPATH=src/GridKIT:src` or it will `ModuleNotFoundError` on `core` or `GridKIT` depending on which file you started from. `scripts/run_experiment.py` and `test_run_training.py` patch `sys.path`/`PYTHONPATH` themselves at the top for this reason.

## Architecture: the `core` hub

```
src/GridKIT/
├── core/          # shared pydantic models, settings, constants, Protocol interfaces
├── grid_model/    # network construction + episode simulation — the only module that owns grid physics
├── map_ui/        # Streamlit + Folium area picker → GridNetwork
├── rl_engine/     # RLlib IPPO training against GridEnvProtocol
├── scenarios/     # runs a rule-based Policy against an injected env → EpisodeMetrics
├── dashboard/     # effectively empty (see below)
└── scripts/       # console-script entry points; several are unimplemented stubs
```

Modules other than `core` do not import each other's internals — `rl_engine` and `scenarios` are written against `core.protocols.GridEnvProtocol` and only ever receive a concrete env (`grid_model.GridEnv`) by dependency injection, never by importing `grid_model` directly for typing/logic. Verified: `scenarios/runner.py` and `scenarios/policies.py` import only from `core`.

- `core/models.py` — the shared data contract: `GridNetwork`/`BusModel`/`LineModel`/`TransformerModel` (topology), `Observation`/`StepResult`/`PowerFlowResult` (per-step env output), `HouseholdDevices` (per-house equipment layout), per-device action enums (`ChargingAction`, `BatteryAction`, `HPAction`), and the agent-id convention `f"{bus_id}::{device}"` via `make_agent_id`/`device_of`/`bus_of`.
- `core/protocols.py` — `GridEnvProtocol` (implemented by `grid_model.GridEnv`; every other module programs only against this) and `NetworkBuilderProtocol` (implemented by `StubNetworkBuilder`/`FixedNetworkBuilder`/`OSMNetworkBuilder`).
- `core/config.py` — one `Settings` (pydantic-settings) singleton, `from core.config import settings`, env-var/`.env`-driven; holds DQN *and* IPPO hyperparameter blocks even though only IPPO is actually implemented (see rl_engine below).
- `core/constants.py` — every tunable number in the project (§14a floor, device ratings, reward weights with reasoning comments for why e.g. the SoC-miss penalty is quadratic, normalization bounds). Read this file before changing behavior elsewhere — most modules just reference these names.

### grid_model — the only module that touches grid physics

- `builder.py` — three `NetworkBuilderProtocol` implementations: `StubNetworkBuilder` (loads a `GridNetwork` from a JSON file, e.g. `data/stub_network.json` or `data/feeder_20.json`), `FixedNetworkBuilder` (wraps an already-built network), `OSMNetworkBuilder` (drives the vendored `vendor/GridCreator` tool for a bounding box). GridCreator runs **out-of-process via `conda run -n GridCreator`** — never imported into GridKIT's own venv, because it needs its own dependency stack and has plotting code at module import time. A `GridCreator` conda env exists on this machine (`conda env list`), so `OSMNetworkBuilder` is actually runnable here, not just documented.
- `gridcreator_loader.py` — a second GridCreator entry point via the pre-downloaded ding0 grid archive (no conda env needed for this path); converts a `pypsa.Network` into `GridNetwork`.
- `network.py` — `build_pypsa_network`: builds a `pypsa.Network` (with a slack generator) from a `GridNetwork`. **Only used as a validation oracle in tests** (`test_network.py`, `test_surrogate.py`) — it is not on the runtime path `GridEnv` actually executes.
- `surrogate.py` — `RadialPowerFlow`: the power flow `GridEnv.step()` actually calls. A fast **linearized DistFlow solver** for one-or-many radial LV feeders (line flow = downstream load sum, voltage ≈ linear drop; loss-free, active-power-only), validated against `build_pypsa_network` in `test_surrogate.py`. Assumes one transformer per feeder component — `OSMNetworkBuilder.build_single_feeder` exists to cut a multi-transformer GridCreator network down to that shape first.
- `environment.py` — `GridEnv(GridEnvProtocol)`. One instance per episode. Every household always draws base load; controllable devices (EV/battery/HP) are each their own agent, configured per household via `HouseholdDevices`, or uniformly via the legacy `ev_penetration=` shortcut. Pure per-device dynamics (`battery_step`, `hp_step`, `curtail_household`) are free functions with no env state, unit-tested directly.
- `device_profiles.py` — real per-household exogenous series (base load, EV availability, HP demand, PV, outdoor temp) from GridCreator's pyCity generators. pyCity simulates a full stochastic year, too slow per-episode, so an annual pool is built once and cached to disk; each episode slices a 24h window and resamples to the 96-step/15-min episode clock.
- `profiles.py` — synthetic fallback profiles (BDEW H0-shaped base load, synthetic day-ahead price) used when a network has no real GridCreator data attached.
- `network_editor.py` — pure `GridNetwork` mutation helpers (`add_bus`, etc.); each returns a new network (`model_copy`) rather than mutating in place, for UI undo/redo.
- `episode_plots.py` — runs one episode end-to-end and saves plots; backs `cli.py plot`.
- `cli.py` — `python -m grid_model.cli {build,plot,add-bus,add-household,add-ev}`; round-trips `GridNetwork` as JSON via `model_dump_json`/`model_validate_json` between invocations so commands chain. Verified working: `build --stub` produces a 5-household network; `plot` runs a full episode against it (curtailment events, peak transformer loading, SoC outcomes, reward all printed) and writes plots.

### map_ui — two independent OSM paths, only one wired to the UI

- `map_widget.py` — the actual Streamlit app (`python -m streamlit run src/GridKIT/map_ui/map_widget.py`). Draw/enter a bounding box → calls `grid_model.builder.OSMNetworkBuilder` (the GridCreator-conda-subprocess path above) → shows the resulting network.
- `osm_fetcher.py` — a second, self-contained network builder (`build_grid_network_from_bounds`) that hits the Overpass API directly with `requests` and parses nodes/ways/buildings itself, with no GridCreator/conda dependency. Exported from `map_ui/__init__.py` and covered by its own tests, but **`map_widget.py` does not call it** — only `AreaBounds` is imported from this file into the widget. Treat it as a standalone/alternate path, not dead code, but don't assume it's what the UI produces.

### rl_engine — only IPPO is implemented

- `ippo_config.py` — `create_ippo_config()` builds a `PPOConfig`: one shared policy **per device type** (`ev_policy`/`battery_policy`/`hp_policy`, not one policy per household and not one global policy), routed via `policy_mapping_fn`. Necessary because EV/battery have 3 discrete actions and HP has 2 — a single shared space would let RLlib sample an invalid HP action.
- `grid_env_rllib_wrapper.py` — `GridEnvRLlibWrapper(MultiAgentEnv)` adapts a `GridEnvProtocol` instance (dependency-injected, not constructed internally) to RLlib's multi-agent interface; per-agent Dict observation/action spaces (heterogeneous action spaces by device type); observations go through `obs_norm.normalize_observation_multidevice`.
- `obs_norm.py` — single source of truth for raw `Observation` → normalized `[0,1]` vector, shared by the training-time wrapper and the eval-time `RLlibPolicyAdapter` so the two stay consistent. Also keeps a legacy 7-dim single-device `normalize_observation` used only by older tests.
- `rl_policy.py` — `RLlibPolicyAdapter`: wraps trained per-device RLModules as a `scenarios.Policy` (grouped batched forward pass per device type; samples from the action distribution by default rather than argmax, so identical agents don't all move in lockstep) so the trained RL policy can be evaluated through the same `scenarios.runner` as the rule-based baselines.
- `trainer.py` — `Trainer`: thin wrapper around Ray init/`register_env`/`PPOConfig.build()`/train-loop/checkpoint/stop. **Verified working**: a `create_ippo_config(num_env_runners=0)` build + one `algo.train()` call against `GridEnv` actually completes and returns a real `episode_return_mean`.
- `callbacks.py` — a `TrainingCallback` Protocol + `DefaultCallback` that prints progress; no RLlib episode-callback integration despite the name.
- `metrics.py`, `local_trainer.py` — empty stubs (metrics.py is a docstring only; local_trainer.py is a 0-byte file).
- `hpc_trainer.py` — docstring-only stub for a future SLURM/bwUniCluster trainer.
- DQN: `core.config.Settings`/`core.constants` define a full DQN hyperparameter block and `algorithm: Literal["dqn","ippo"]`, but **no DQN implementation exists anywhere in the codebase** — only the IPPO path above is real.

### scenarios

- `policies.py` — rule-based, non-learning household behaviours implementing the `Policy` protocol (`reset`/`act`), imports only `core`: `NaiveImmediatePolicy` (charge EV immediately, flat tariff) and `NaivePriceFollowPolicy` (EV waits for the cheapest contiguous price window; `jitter_std` controls how synchronized households are — 0 = every EV picks the identical window). Battery is always greedy self-consumption, HP always thermostatic, in both.
- `runner.py` — drives a `Policy` against an injected `GridEnvProtocol` over seeds and reduces the per-step stream to `EpisodeMetrics`/`SimResult` (mean±std across seeds). The trained RL policy (via `rl_engine.RLlibPolicyAdapter`) runs through this exact same runner as the rule-based baselines for a like-for-like comparison — see `scripts/run_experiment.py`.

### scripts and dashboard — mixed implemented/stub

- `run_experiment.py` — the one fully wired end-to-end script: for each EV penetration level in `core.constants.EV_PENETRATION_LEVELS`, trains IPPO, then evaluates it plus both rule-based baselines via `scenarios.runner`, writing `summary.json`/`timelines.json`/checkpoints. Patches `sys.path`/cwd itself at the top (see the two-import-styles note above).
- `plot_results.py` — reads `run_experiment.py`'s output and produces plots (not exercised here; see the file directly for details).
- `app.py:main`, `train.py:main` — both `raise NotImplementedError`; not usable despite being registered as `gridkit-app`/`gridkit-train` console scripts in `pyproject.toml`.
- `dashboard/` — effectively empty: `__init__.py` is 0 bytes. A `report.py` existed previously (a stale `.pyc` remains in `__pycache__`) but has been deleted from the working tree.

## Gitignored / regenerate-locally paths

`outputs/`, `outputs_*/`, `checkpoints/`, `cache/`, `data/oberacker.json`, `.venv/` — don't expect these to be present, and don't commit into them.
