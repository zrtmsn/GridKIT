# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Environment is `uv`-managed; the venv is `.venv` (Python ≥3.12).

```bash
uv sync                                     # install deps + create .venv

.venv/bin/python -m pytest -q               # full suite (~128 tests)
.venv/bin/python -m pytest src/GridKIT/grid_model/test_environment.py -q       # one file
.venv/bin/python -m pytest src/GridKIT/grid_model/test_environment.py::test_three_agents_per_active_household   # one test
```

`pyproject.toml` sets `pythonpath = ["src/GridKIT", "src"]`, so pytest needs no `PYTHONPATH`. Anything run outside pytest does: either `PYTHONPATH=src` (for `GridKIT.*` imports) or `PYTHONPATH=src/GridKIT:src`. Scripts under `scripts/` bootstrap `sys.path` themselves and `os.chdir` to the repo root.

```bash
# full experiment: train IPPO per EV-penetration level, evaluate scenarios 1–3 → outputs/
PYTHONPATH=src .venv/bin/python -m GridKIT.scripts.run_experiment --iterations 40 --seeds 12 \
    --network data/feeder_20.json --out outputs
# smoke version: --iterations 2 --seeds 2 --network data/stub_network.json --out outputs_smoke

PYTHONPATH=src .venv/bin/python -m GridKIT.scripts.run_scenarios --seeds 6   # non-RL scenarios only
PYTHONPATH=src .venv/bin/python -m GridKIT.scripts.fetch_network --south .. --west .. --north .. --east .. --out data/x.json

PYTHONPATH=src .venv/bin/streamlit run src/GridKIT/scripts/webapp.py         # design + background training + results
PYTHONPATH=src .venv/bin/streamlit run src/GridKIT/dashboard/app.py          # results only (reads $GRIDKIT_OUTPUT_DIR, default outputs/)
gridkit-app --train                                                          # run experiment, then open dashboard
```

Console scripts (`pyproject.toml`): `gridkit-app`, `gridkit-train`, `gridkit-experiment`.

## Architecture

GridKIT simulates a low-voltage distribution feeder where every household's flexible devices are RL agents, to compare naive charging behaviours against a learned policy under §14a EnWG curtailment.

### The one hard rule

Modules never import from each other — only from `core`. `core/models.py`, `core/constants.py`, `core/protocols.py`, `core/config.py` are the shared language. Cross-module wiring happens only in composition roots (`scripts/`, and `scripts/grid_designer.py` which says so explicitly). Artem owns and reviews `core`.

```
map_ui ─┐
grid_model ─┼─► core ◄─ scenarios
rl_engine ─┘        ◄─ dashboard
```

Consequence in practice: `scenarios/runner.py` takes an injected `GridEnvProtocol`, `GridEnv` takes an injected builder and `DeviceProfileProvider`, `Trainer` takes an `env_factory` callable.

### Import-path gotcha

Because both `src/` and `src/GridKIT/` are on the path, the same module is importable under two names: `core.constants` and `GridKIT.core.constants`. `grid_model/`, `scenarios/`, `map_ui/`, `scripts/` use the short form; `rl_engine/` and top-level `test_run_training*.py` use the `GridKIT.` form. Both work, but a module imported under both names gets **two separate instances** — never rely on module-level mutable state across the boundary, and match the surrounding file's convention when editing.

### Agents and the device model

One controllable device = one agent. `agent_id = f"{bus_id}::{device}"` (`AGENT_SEP`, `make_agent_id`/`bus_of`/`device_of` in `core/models.py`). Devices: `ev`, `battery`, `hp`. PV is exogenous generation folded into the meter, **not** an agent. Each device type has one shared IPPO policy (`ev_policy` / `battery_policy` / `hp_policy`), routed by `ippo_config.policy_mapping_fn` on the id suffix. Action spaces differ per type (ev/battery `Discrete(3)`, hp `Discrete(2)`).

All device-agents in one house share a single household reward (`GridEnv._reward`): net electricity bill + EV SoC shortfall + HP comfort. Rewards are denominated in **euros** — one unit of reward = one euro — and the constants comments in `core/constants.py` record why each weight has its value (the EV shortfall penalty is quadratic because linear was measured to be exploitable). Read those comments before changing a reward constant.

Household equipment is a `HouseholdDevices` layout (`build_device_layout`); passing only `ev_penetration` to `GridEnv` builds the legacy joint layout where the same fraction of homes gets the whole stack.

### Episode and physics

96 steps × 15 min = 24 h, starting at `EPISODE_START_HOUR = 12` so the overnight charging window never wraps. Power flow runs every step via `grid_model/surrogate.py` `RadialPowerFlow` — a linearized solver over a *forest* of radial feeders (exact downstream-sum line flows + DistFlow voltages), validated against PyPSA `pf` in `test_surrogate.py`. PyPSA (`grid_model/network.py`) is the slow validation path only.

`PowerFlowResult.transformer_loading_pu` is the **max feeder loading**, not one transformer; per-feeder detail is in `transformer_loadings_pu`. Cables, not transformers, are typically the binding constraint on real ding0 grids — always report peak line loading alongside the transformer figure.

§14a curtailment is **environment physics, not a reward term**: when a feeder or a household's path line exceeds threshold, that household's controllable power is dimmed proportionally with a `MIN_GUARANTEED_POWER_KW` floor, then power flow is re-solved. Households on healthy feeders are untouched. Agents perceive congestion only through *lagged* observations (`local_voltage_pu`, `recent_curtailment_ratio`) — that is the whole mechanism by which a selfish policy learns to avoid stress.

### Exogenous profiles

`grid_model/device_profiles.py` generates base load, EV availability, HP demand, PV and outdoor temperature from GridCreator/pyCity with bundled TRY weather. A full stochastic year is simulated once, cached to `cache/device_profiles_<hash>.npz`, and each episode slices a noon-to-noon day. **One weather day is drawn per feeder per episode**, so seed-to-seed variance is dominated by weather, not by EV behaviour — use ≥50 seeds for any claim about curtailment. Only the day-ahead price stays synthetic (`grid_model/profiles.py`).

### Networks

`data/stub_network.json` (5 households, deliberately small and stressable) and `data/feeder_20.json` (4×5 households) are the built-in test grids. Real topology comes from either `map_ui/osm_fetcher.py` (OSM — one generic 160 kVA transformer, so its congestion numbers are not physically meaningful) or `grid_model/gridcreator_loader.py` (ding0 archive — real, individually-sized transformers). The ding0 archive is optional; see README for the Zenodo download and `$GRIDKIT_DING0_GRIDS_DIR`.

### Runs

`scripts/webapp.py` → `run_store.py` creates `runs/<id>/` (config, network, layout, status, results, checkpoints) and launches `scripts/train_run.py` as a detached subprocess that updates `status.json` live. `run_store.py` is filesystem-only and root-configurable so it is testable without Streamlit or Ray.

## Testing conventions

Tests are **colocated** next to the module (`grid_model/test_environment.py`, not a `tests/` directory — the README is out of date on this). Use the `_provider()` helper in `grid_model/test_environment.py` for a synthetic `DeviceProfileProvider` with constant profiles: it makes tests deterministic and skips pyCity entirely. Other test files import it directly.

## Reference docs

`md/` holds design notes worth reading before large changes: `gridkit_architecture.md`, `gridkit_core_models.md`, `gridkit_math_formulation.md`, `gridkit_rl_decisions.md`, `gridkit_rl_training_loop.md`, `gridkit_price_reactive_results.md`.

## Known inconsistencies

- `OBS_DIM = 7` (EV-only legacy) vs `OBS_DIM_MULTIDEVICE = 10`; `Observation` carries the multi-device fields and `rl_engine/obs_norm.py` has both normalizers.
- `settings.algorithm` defaults to `"dqn"` and DQN constants exist, but there is no DQN implementation — IPPO is the only trainer.
- `gridcreator_loader`'s `residential_only` filters on `type`, which every ding0 load shares; the real discriminator is the `sector` column.
