# GridKIT — RL Design Decisions

> Decisions made during conceptual design phase. Last updated: May 2026.

---

## Project Context

GridKIT is a simulator for low-voltage distribution networks (based on real OSM topology of Karlsruhe) with multi-agent reinforcement learning. The primary use case is a planning tool for grid operators to assess how smart EV charging management affects grid stability under §14a EnWG.

---

## 1. Who is the Agent?

**Decision: Agent = Household**

Each household managing one or more controllable devices acts as an independent agent. The grid operator is part of the environment, not an agent.

**Rationale:** This reflects the real-world structure of §14a — households participate in demand response, and the grid operator enforces curtailment rules. Households do not coordinate with each other.

---

## 2. Action Space

**Decision: Discrete(3) — three charging power levels**

```python
class ChargingAction(IntEnum):
    OFF  = 0  # 0.0 kW
    HALF = 1  # 3.7 kW
    FULL = 2  # 7.4 kW

ACTION_TO_KW = {
    ChargingAction.OFF:  0.0,
    ChargingAction.HALF: 3.7,
    ChargingAction.FULL: 7.4,
}
```

**Rationale:** Three actions are sufficient for meaningful behaviour (do not charge during peak / charge slowly / charge at full speed). Discrete action space enables DQN, which is simpler to train and debug than continuous alternatives (SAC, PPO with continuous head). The action space can be extended later without changing the rest of the codebase.

Gymnasium interface: `spaces.Discrete(3)`

---

## 3. Reward Structure

**Decision: Task-completion based, not penalty-based**

```
r = + progress_toward_soc_target      # positive signal each timestep
  + large_bonus_if_target_reached     # terminal bonus at departure time
  - large_penalty_if_target_missed    # terminal penalty at departure time
  - electricity_cost_weight * kWh     # small cost signal each timestep
```

**Key design choice:** §14a curtailment events are NOT a separate penalty term. When the environment curtails the agent to 4.2 kW, the agent fails to charge as requested — this propagates naturally into the task-completion component. No explicit curtailment penalty is needed.

**Rationale:** Inaction (action = OFF always) must be worse than attempting to charge and being curtailed. The terminal penalty for missing the departure SoC target ensures the agent is motivated to act. The agent learns to avoid curtailment implicitly by learning to charge when the grid is not overloaded.

---

## 4. Observation Space

**Decision: Local observations only (no grid-level information)**

Each agent observes only what a smart meter at its own household can measure:

```python
observation = [
    current_soc / target_soc,              # normalised progress toward goal
    time_until_departure / episode_length, # urgency signal
    current_electricity_price,             # tariff for this timestep
    base_load_kw,                          # current inflexible household consumption
    outdoor_temperature_c,                 # context for future phases (heat pump)
]
```

**What agents do NOT see:**
- Actions or state of neighbouring households
- Transformer load or line utilisation
- Whether other agents are being curtailed

**Rationale:** This is realistic — smart meters only observe local data. It is also the interesting research baseline: the research question is how much performance is lost compared to a centralised or partially observable variant. This can be explored in a later phase.

---

## 5. Agent Behaviour Model

**Decision: Selfish (non-cooperative) agents**

Each agent optimises only its own reward. There is no shared reward component, no explicit coordination mechanism, and no communication between agents.

**Rationale:** This is the most realistic starting point — households in the real world do not coordinate their charging. It also produces the worst-case grid loading scenario, which is exactly what the grid operator wants to stress-test. Cooperative variants can be added later as a comparison.

---

## 6. Devices in Scope (Phase 1)

**Decision: Electric vehicle (EV) charging only**

One EV charger (wallbox) per household. Maximum power: 7.4 kW (32A single-phase). Minimum charging power when active: 1.4 kW (6A).

**Out of scope for Phase 1:**
- Heat pumps (require thermal dynamics model)
- Battery storage (requires bidirectional power flow)
- PV generation (requires solar irradiance data)

**Rationale:** EV charging has a single state variable (State of Charge, 0–100%) and a clear deadline (departure time). This makes the reward function simple and the task well-defined. Adding further device types is straightforward once the core loop is validated.

---

## 7. Episode Structure

- **Timestep:** 15 minutes
- **Episode length:** 96 timesteps = 24 hours
- **Reset:** Each episode samples a new scenario from the data distributions below

At `reset()`, the environment samples:
```
arrival_time   ~ N(18:00, 1h)       # EV arrives home
initial_soc    ~ N(0.30, 0.10)      # battery level on arrival
departure_time ~ N(07:00, 0.5h)     # EV must be ready by this time
target_soc     = 0.80               # fixed target (can be randomised later)
load_profile   = BDEW_H0[random_day]
price_profile  = ENTSO_E[random_day]
```

---

## 8. Data Sources

Data is used to parametrise the simulation environment — it is not a training dataset.

| Data | Source | Purpose |
|---|---|---|
| Grid topology | OpenStreetMap via GridCreator | Network graph, line parameters, transformer ratings |
| Base load profile | BDEW H0 standard profile | Inflexible household consumption per 15-min slot |
| Electricity prices | ENTSO-E Transparency Platform (API) | Price signal in observation and cost component of reward |
| Outdoor temperature | DWD (Deutscher Wetterdienst) | Environment context; required for heat pump phase |
| EV usage patterns | MiD-Studie (Mobilität in Deutschland) | Sample arrival/departure times and initial SoC |

**Phase 1 minimum:** BDEW H0 profile + synthetic EV schedules are sufficient to run the first training loop. Real price data and temperature can be added incrementally.

---

## 9. Generalisation Strategy

**Decision: Train on a distribution of scenarios, not a single year of data**

To avoid the agent overfitting to one set of prices or load patterns, training episodes sample from a range of conditions:

```python
load_multiplier  = Uniform(0.8, 1.4)   # scale base load up/down
price_scenario   = Choice(["low", "medium", "high"])
ev_penetration   = fixed per experiment (20% / 40% / 60% of households have EVs)
```

**Rationale:** The grid operator wants to understand sensitivity to parameters, not just performance under 2024 conditions. Running experiments across EV penetration levels (20% → 40% → 60%) is the primary analytical output of the tool.

If conditions change significantly (new tariff structure, updated load profiles), retraining takes a few hours and is the intended workflow — this is a planning tool, not a continuously deployed system.

---

## 10. Algorithm

**Decision: DQN for Phase 1**

Deep Q-Network with discrete action space (3 actions). Independent learners — each agent runs its own DQN with no parameter sharing in Phase 1.

**Rationale:** DQN is well-understood, easy to debug, and compatible with `spaces.Discrete(3)`. The independent learner assumption is consistent with the selfish agent decision above.

Future phases may explore:
- Parameter sharing across agents (faster convergence, implicit coordination)
- QMIX or VDN for cooperative variants
- Continuous action space with SAC

---

## 11. Value for Grid Operators

The tool answers the question:

> "Given our actual network topology and a forecast EV penetration rate, how often will §14a curtailment events occur under selfish charging behaviour — and does smart RL-based scheduling reduce this without requiring explicit coordination?"

Key outputs from a simulation run:
- Number of §14a interventions per day at each EV penetration level
- Transformer and line loading statistics
- SoC satisfaction rate (% of agents meeting departure target)
- Comparison: uncontrolled charging vs RL agent vs naive time-shifting baseline
