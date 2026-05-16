# GridKIT — Core Models Reference

> Who produces each model, who consumes it, and what it represents. Last updated: May 2026.

---

## Enums

### `ChargingAction`
The three decisions an agent can make: OFF / HALF / FULL. rl_engine produces one per agent per timestep and passes the dict to `env.step()`. grid_model receives it, looks up the kW value via `ACTION_TO_KW`, and applies it to the power flow.

### `PriceScenario`
LOW / MEDIUM / HIGH. Not used in Phase 1 — placeholder for when grid_model starts sampling synthetic price regimes at `reset()`.

---

## Topology

### `BusModel`, `LineModel`, `TransformerModel`
Building blocks of `GridNetwork`. Only assembled by grid_model's network builder — nobody instantiates them directly elsewhere.

### `GridNetwork`
The assembled network topology. Produced once by grid_model's network builder (Phase 1: `StubNetworkBuilder` loads from stub JSON; Phase 2: `OSMNetworkBuilder` runs GridCreator). Lives for the entire run — never changes after construction.

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
The 5-dimensional vector the agent learns from. grid_model builds one per agent inside `step()` by reading the current `EVState` and episode data. rl_engine's `HouseholdAgent` calls `obs.to_array()` to get a numpy-ready list, feeds it into the DQN, and stores it in the replay buffer.

### `StepResult`
Wraps everything grid_model returns for one agent from one timestep: the next `Observation`, the `reward`, and the `done` flag. rl_engine's Trainer unpacks it — passes `observation` back into the agent for the next step, passes `reward` and `done` to `store_transition()`.

These two are the core handshake between grid_model and rl_engine. Everything else is either setup or logging.

---

## Power Flow

### `PowerFlowResult`
Produced by grid_model once per timestep — one object shared across all agents (network-level result, not per-agent). Returned from `step()` as the second element of the tuple alongside the per-agent `StepResult` dict.

rl_engine's Trainer collects one per timestep and appends it to a list. At episode end it uses that list to build `SimResult.timestep_results`. grid_model also uses it internally to decide curtailment before returning.

---

## Episode Results

### `EpisodeMetrics`
Computed by rl_engine's Trainer at the end of each episode. Aggregates: mean reward, fraction of agents that hit their SoC target, curtailment event count, peak transformer load, and current epsilon. Consumed by dashboard for training progress charts.

### `SimResult`
Full episode record, assembled by Trainer. Combines the `list[PowerFlowResult]` collected during the episode, `final_soc_per_agent` (SoC of each agent at departure), and `EpisodeMetrics`. dashboard reads `SimResult` to render network loading over time, per-agent SoC outcomes, and curtailment heatmaps.

---

## Summary

| Model | Produced by | Consumed by |
|---|---|---|
| `ChargingAction` | rl_engine | grid_model |
| `GridNetwork` | grid_model (builder) | grid_model, rl_engine, dashboard |
| `EVState` | grid_model | grid_model only |
| `Observation` | grid_model | rl_engine |
| `StepResult` | grid_model | rl_engine |
| `PowerFlowResult` | grid_model | rl_engine (→ SimResult), dashboard |
| `EpisodeMetrics` | rl_engine | dashboard |
| `SimResult` | rl_engine (Trainer) | dashboard |
