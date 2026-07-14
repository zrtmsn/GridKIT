# GridKIT — Core Models Reference

> Who produces each model, who consumes it, and what it represents. Last updated: June 2026.

---

## Enums

### `ChargingAction`
The three decisions an agent can make: OFF / HALF / FULL. rl_engine (the shared IPPO policy, via `GridEnvRLlibWrapper`) produces one per agent per timestep and passes the dict to `env.step()`. grid_model receives it, looks up the kW value via `ACTION_TO_KW`, and applies it to the power flow.

### `PriceScenario`
LOW / MEDIUM / HIGH. Defined but **not yet wired in** — placeholder for when grid_model starts sampling price regimes at `reset()`. Currently price is a flat hardcoded constant in `environment.py`.

---

## Topology

### `BusModel`, `LineModel`, `TransformerModel`
Building blocks of `GridNetwork`. Only assembled by network builders — nobody instantiates them directly elsewhere.

### `GridNetwork`
The assembled network topology. Two builders currently exist: `grid_model.StubNetworkBuilder` (loads a fixed stub JSON, used by `GridEnv` today) and `map_ui.build_grid_network_from_bounds()` (real OSM topology via Overpass, implemented but **not yet connected to `GridEnv`**). Lives for the entire run — never changes after construction.

| Consumer | What it reads |
|---|---|
| grid_model | buses/lines/transformers to build the internal PyPSA network |
| rl_engine | `household_bus_ids` to know how many agents to create and which bus each sits on |
| dashboard | topology for network visualization |

---

## EV State

### `EVState`
**Internal to grid_model only.** grid_model maintains one `EVState` per agent and updates it every timestep — increments `soc` based on actual power delivered after curtailment, flips `is_connected` when the EV arrives or departs. Never passed to rl_engine. grid_model uses it to compute `Observation`.

---

## RL Interface

### `Observation`
The 5-dimensional vector the agent learns from. grid_model builds one per agent inside `step()` by reading the current `EVState` and episode data. `GridEnvRLlibWrapper` calls `obs.to_array()` to convert it into RLlib's numpy/dict format for the shared `household_policy` (IPPO via Ray RLlib). Note: `electricity_price` is currently a raw, unnormalised value — see open issue in `gridkit_rl_decisions.md` §4.

### `StepResult`
Wraps everything grid_model returns for one agent from one timestep: the next `Observation`, the `reward`, and the `done` flag. `GridEnvRLlibWrapper` unpacks it into the RLlib step-return format Ray expects.

These two are the core handshake between grid_model and rl_engine. Everything else is either setup or logging.

---

## Power Flow

### `PowerFlowResult`
Produced by grid_model once per timestep — one object shared across all agents (network-level result, not per-agent). Returned from `step()` as part of the info/result the wrapper passes through.

grid_model uses it internally to decide curtailment before returning. **Not yet collected/aggregated anywhere** — there's no episode-level accumulation into `SimResult` yet.

---

## Episode Results — not yet implemented

### `EpisodeMetrics`
Defined in `core/models.py` (mean reward, SoC target hit rate, curtailment event count, peak transformer load). **Nothing in rl_engine currently populates this** — `rl_engine/metrics.py` is a stub. This is the model that a baseline-vs-RL comparison would need to fill in (see `gridkit_rl_decisions.md` §11).

### `SimResult`
Defined in `core/models.py` (per-timestep `PowerFlowResult` list, final SoC per agent, `EpisodeMetrics`). **Not yet assembled by anything.** `dashboard` is an empty package, so there's currently no consumer either.

---

## Summary

| Model | Produced by | Consumed by | Status |
|---|---|---|---|
| `ChargingAction` | rl_engine (shared IPPO policy) | grid_model | implemented |
| `GridNetwork` | grid_model (`StubNetworkBuilder`) or map_ui (`build_grid_network_from_bounds`) | grid_model, rl_engine | implemented; map_ui path not yet connected to `GridEnv` |
| `EVState` | grid_model | grid_model only | implemented |
| `Observation` | grid_model | rl_engine | implemented; price field unnormalised |
| `StepResult` | grid_model | rl_engine | implemented |
| `PowerFlowResult` | grid_model | (internal only currently) | implemented, not yet aggregated |
| `EpisodeMetrics` | — | — | **not implemented** |
| `SimResult` | — | — | **not implemented** |
