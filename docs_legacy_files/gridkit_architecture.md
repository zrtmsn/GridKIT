# GridKIT — Module Architecture

> How grid_model, rl_engine, map_ui, and dashboard communicate through core models. Last updated: June 2026.

---

```mermaid
flowchart TD
    subgraph core ["core/models.py · single source of truth"]
        GridNetwork
        Observation
        ChargingAction
        StepResult
        PowerFlowResult
        EpisodeMetrics
        SimResult
    end

    subgraph grid_model ["grid_model"]
        builder["StubNetworkBuilder\n.build()"]
        env["GridEnv\n.reset() / .step()\n(implements GridEnvProtocol)"]
        pypsa(["PyPSA power flow\n+ §14a curtailment"])
        evstate["EVState × N\n(internal only)"]
    end

    subgraph rl_engine ["rl_engine"]
        wrapper["GridEnvRLlibWrapper\n(DI wrapper around GridEnvProtocol)"]
        trainer["Trainer\n(Ray RLlib training loop)"]
        policy["household_policy\nshared PPO policy (IPPO)"]
    end

    subgraph map_ui ["map_ui — implemented, not yet wired into grid_model"]
        map["build_grid_network_from_bounds()\nOverpass/OSM → GridNetwork\n+ map_widget visualisation"]
    end

    subgraph dashboard ["dashboard — not yet implemented"]
        ui["Streamlit UI\n(planned: training charts · SoC · curtailment heatmaps)"]
    end

    %% ── SETUP ────────────────────────────────────────────────
    builder -->|"GridNetwork"| env
    map -.->|"GridNetwork (not yet connected)"| env

    %% ── EPISODE RESET ────────────────────────────────────────
    trainer -- "reset()" --> wrapper
    wrapper -- "reset()" --> env
    env -->|"dict[agent_id → Observation]"| wrapper

    %% ── TIMESTEP LOOP ×96 ────────────────────────────────────
    wrapper -->|"obs.to_array() (RLlib dict/numpy format)"| policy
    policy -->|"ChargingAction"| wrapper
    wrapper -- "step(dict[id → ChargingAction])" --> env
    env <-->|"loads / voltages"| pypsa
    pypsa --> evstate
    evstate -->|"next Observation + reward"| env
    env -->|"dict[id → StepResult]"| wrapper
    env -->|"PowerFlowResult"| wrapper

    %% ── LEARNING ─────────────────────────────────────────────
    trainer -->|"PPO update (Ray RLlib)"| policy

    %% ── EPISODE END ──────────────────────────────────────────
    trainer -.->|"EpisodeMetrics (not yet populated)"| ui
    trainer -.->|"SimResult (not yet populated)"| ui
```

---

## Core model ownership

| Model | Produced by | Consumed by | When |
|---|---|---|---|
| `GridNetwork` | grid_model (builder) | grid_model, rl_engine, dashboard, map_ui | setup — once per run |
| `Observation` | grid_model | rl_engine | every timestep, per agent |
| `ChargingAction` | rl_engine | grid_model | every timestep, per agent |
| `StepResult` | grid_model | rl_engine | every timestep, per agent |
| `PowerFlowResult` | grid_model | rl_engine → SimResult | every timestep, shared |
| `EpisodeMetrics` | rl_engine (planned) | dashboard (planned) | episode end — **not yet implemented**, fields defined in core but never populated |
| `SimResult` | rl_engine (planned) | dashboard (planned) | episode end — **not yet implemented** |
| `EVState` | grid_model | grid_model only | internal — never crosses module boundary |

---

## Key design rule

Modules communicate **only through `core/models.py`**. No module imports from another module directly.
`grid_model` implements `GridEnvProtocol`; `rl_engine` depends only on that protocol interface, never on the concrete implementation (`GridEnvRLlibWrapper` is constructed with an injected `GridEnvProtocol` instance).

## Current gaps (June 2026)

- `map_ui` produces a real-OSM `GridNetwork` but `grid_model.GridEnv` still only builds from `StubNetworkBuilder` — the two are not yet connected.
- `dashboard` has no implementation beyond an empty package — `EpisodeMetrics`/`SimResult` are defined in `core/models.py` but nothing in `rl_engine` populates them yet.
- No baseline/naive policy or evaluation harness exists — see `gridkit_rl_decisions.md` §11.
