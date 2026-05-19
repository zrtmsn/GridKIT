# GridKIT — Module Architecture

> How grid_model, rl_engine, map_ui, and dashboard communicate through core models. Last updated: May 2026.

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
        env["GridKITEnv\n.reset() / .step()"]
        pypsa(["PyPSA power flow\n+ §14a curtailment"])
        evstate["EVState × N\n(internal only)"]
    end

    subgraph rl_engine ["rl_engine"]
        trainer["Trainer\n(episode loop)"]
        agent["HouseholdAgent × N\nDQNetwork · ReplayBuffer"]
    end

    subgraph map_ui ["map_ui"]
        map["Network topology\nvisualisation"]
    end

    subgraph dashboard ["dashboard"]
        ui["Streamlit UI\ntraining charts · SoC · curtailment heatmaps"]
    end

    %% ── SETUP ────────────────────────────────────────────────
    builder -->|"GridNetwork"| env
    builder -->|"GridNetwork"| trainer
    builder -->|"GridNetwork"| map
    builder -->|"GridNetwork"| ui

    %% ── EPISODE RESET ────────────────────────────────────────
    trainer -- "reset()" --> env
    env -->|"dict[agent_id → Observation]"| trainer

    %% ── TIMESTEP LOOP ×96 ────────────────────────────────────
    trainer -->|"Observation"| agent
    agent -->|"ChargingAction"| trainer
    trainer -- "step(dict[id → ChargingAction])" --> env
    env <-->|"loads / voltages"| pypsa
    pypsa --> evstate
    evstate -->|"next Observation + reward"| env
    env -->|"dict[id → StepResult]"| trainer
    env -->|"PowerFlowResult"| trainer

    %% ── LEARNING ─────────────────────────────────────────────
    trainer -->|"store_transition + TD update"| agent

    %% ── EPISODE END ──────────────────────────────────────────
    trainer -->|"EpisodeMetrics"| ui
    trainer -->|"SimResult"| ui
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
| `EpisodeMetrics` | rl_engine | dashboard | episode end |
| `SimResult` | rl_engine | dashboard | episode end |
| `EVState` | grid_model | grid_model only | internal — never crosses module boundary |

---

## Key design rule

Modules communicate **only through `core/models.py`**. No module imports from another module directly.
`grid_model` implements `GridEnvProtocol`; `rl_engine` depends only on that protocol interface, never on the concrete implementation.
