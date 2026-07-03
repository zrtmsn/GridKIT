# GridKIT — RL Design Decisions

> Decisions made during conceptual design and revised after implementation. Last updated: June 2026.

---

## Core Thesis

We are not testing a grid-friendly agent. We are testing whether §14a's blunt curtailment
mechanism holds up against realistic, selfish, price-reactive customers — and whether smarter
(RL) selfish behaviour makes that better or worse, e.g. by avoiding synchronised charging spikes
that naive price-following would create.

The agent is designed as a self-interested customer (charge by deadline, minimise own cost). It
is never rewarded for grid stability. Grid stability is an emergent outcome of many such agents
interacting with the physical network and §14a's enforcement — that outcome is what the
simulation measures, not what the agent is told to optimise for.

---

## Project Context

GridKIT is a simulator for low-voltage distribution networks (based on real OSM topology of
Karlsruhe) with multi-agent reinforcement learning. The primary use case is a planning tool for
grid operators to assess how smart EV charging management affects grid stability under §14a EnWG,
under an assumed future of higher electrification and volatile (dynamic) electricity pricing.

---

## 1. Who is the Agent?

**Decision: Agent = Household**

Each household managing one EV charger acts as an independent agent. The grid operator is part of
the environment (it enforces §14a curtailment), not an agent.

**Rationale:** Reflects the real-world structure of §14a — households respond to price/curtailment
individually; the grid operator enforces rules. Households do not coordinate with each other.

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

**Rationale:** Three actions are sufficient for meaningful behaviour (don't charge during peak /
charge slowly / charge at full speed). Discrete action space keeps the policy simple to train and
inspect. Can be extended later without changing the rest of the codebase.

Gymnasium interface: `spaces.Discrete(3)` — **status: implemented**, matches code.

---

## 3. Reward Structure

**Decision: Task-completion based, not penalty-based**

```
r = + progress_toward_soc_target      # positive signal each timestep
  + large_bonus_if_target_reached     # terminal bonus at departure time
  - large_penalty_if_target_missed    # terminal penalty at departure time
  - electricity_cost_weight * kWh     # small cost signal each timestep
```

**Key design choice:** §14a curtailment events are NOT a separate penalty term. When the
environment curtails the agent below its requested power, it simply fails to deliver as much
energy — this propagates into the task-completion term on its own. No explicit curtailment
penalty is needed.

**Rationale:** This keeps the agent's incentives purely customer-side (charge the car, minimise
cost) — consistent with the Core Thesis above. The agent has no reason to "care" about the grid;
any grid-protective behaviour that emerges must come from learning to avoid the conditions that
cause curtailment, not from being told to.

**Status: implemented**, matches `environment.py`.

---

## 4. Observation Space

**Decision: Local observations only (no grid-level information)**

```python
observation = [
    current_soc / target_soc,              # normalised progress toward goal
    time_until_departure / episode_length, # urgency signal
    current_electricity_price,             # tariff for this timestep — currently RAW, unnormalised
    base_load_kw,                          # current inflexible household consumption
    outdoor_temperature_c,                 # context for future phases (heat pump)
]
```

**What agents do NOT see:**
- Actions or state of neighbouring households
- Transformer load or line utilisation
- Whether other agents are being curtailed

**Rationale:** Realistic — smart meters only observe local data. It is also the interesting
research baseline: how much performance is lost compared to a centralised or partially-observable
variant. Can be explored in a later phase.

**Known issue:** price is fed as a raw €/kWh value with no normalisation against the day's
price range or a `PriceScenario` category. This risks the policy learning "cheap" only relative to
whatever range it saw during training, with no principled notion of scale. Candidate fix:
normalise against the known day-ahead price curve (percentile or min-max), since day-ahead prices
are published in advance — this is realistic, not a simplification. Not yet implemented.

**Status: partially implemented** — vector shape matches `Observation.to_array()`; price is raw,
base load/price/temperature are currently flat constants rather than sampled data (see §8).

---

## 5. Agent Behaviour Model

**Decision: Selfish (non-cooperative) reward, shared policy weights**

Each agent optimises only its own reward — no shared reward term, no explicit coordination, no
communication between agents. This part is unchanged from the original design.

**Revised from original design:** the original plan was *fully independent learners* (separate
network per agent, no parameter sharing). The implementation uses **IPPO with one shared policy**
(`household_policy`) across all agents, for training efficiency.

This changes what is being tested, and should be stated explicitly to anyone reading this doc:
- *Independent learners* (original plan) → models "every household learns its own idiosyncratic
  strategy" — a true worst-case, least-coordinated scenario.
- *Shared policy* (actual implementation) → models "every household adopts the same smart-charging
  strategy" (e.g. all customers on the same utility app/optimiser). Agents remain behaviourally
  selfish (reward is individual), but architecturally homogeneous — which can itself cause
  synchronised charging spikes, since identical policies facing similar local signals (e.g. the
  same cheap price window) tend to act alike. This is directly relevant to the Core Thesis: a
  shared-policy population is arguably the *more* realistic stress test of §14a under future
  dynamic pricing, not a weaker one.

**Status: implemented as shared-policy IPPO.** Independent-learner variant is a valid follow-up
experiment, not yet built.

---

## 6. Devices in Scope (Phase 1)

**Decision: Electric vehicle (EV) charging only**

One EV charger (wallbox) per household. Maximum power: 7.4 kW (32A single-phase). Minimum charging
power when active: 1.4 kW (6A).

**Out of scope for Phase 1:** heat pumps, battery storage, PV generation.

**Rationale:** EV charging has a single state variable (SoC) and a clear deadline (departure
time), keeping the reward function simple and well-defined.

**Status: implemented** — EV charging only, no other devices.

---

## 7. Episode Structure

- **Timestep:** 15 minutes
- **Episode length:** 96 timesteps = 24 hours
- **Reset:** intended to sample a new scenario per episode (see distributions below)

```
arrival_time   ~ N(18:00, 1h)       # design target — NOT IMPLEMENTED
initial_soc    ~ N(0.30, 0.10)      # design target — NOT IMPLEMENTED
departure_time ~ N(07:00, 0.5h)     # design target — NOT IMPLEMENTED
target_soc     = 0.80               # implemented (fixed)
load_profile   = BDEW_H0[random_day]   # design target — NOT IMPLEMENTED
price_profile  = ENTSO_E[random_day]   # design target — NOT IMPLEMENTED
```

**Status: not implemented.** Current code hardcodes `arrival_step=0`, `departure_step=95` (full
day availability) due to an unresolved `model_validator` issue, and base load / price are flat
constants, not sampled. Every episode is currently identical — there is no day-to-day or seasonal
variation yet. This is the single biggest gap between the design doc and what can currently be
demonstrated.

---

## 8. Data Sources

Data is used to parametrise the simulation environment — it is not a training dataset.

| Data | Source | Purpose | Status |
|---|---|---|---|
| Grid topology | OpenStreetMap via `map_ui` (Overpass) | Network graph, line parameters, transformer ratings | **Implemented**, not yet wired into `grid_model.GridEnv` (still uses `StubNetworkBuilder`) |
| Base load profile | BDEW H0 standard profile | Inflexible household consumption per 15-min slot | Not implemented (flat constant) |
| Electricity prices | ENTSO-E Transparency Platform (API) | Price signal in observation + cost term | Not implemented (flat constant) |
| Outdoor temperature | DWD (Deutscher Wetterdienst) | Context; required for future heat pump phase | Not implemented |
| EV usage patterns | MiD-Studie (Mobilität in Deutschland) | Sample arrival/departure/initial SoC | Not implemented |

**Note:** this table previously implied "Phase 1 minimum: BDEW H0 + synthetic EV schedules is
sufficient to run the first training loop." In fact the first training loop runs without any of
these — all are still placeholders. Wiring even one of these in (price is highest priority, see
Core Thesis) is the next concrete step, not a "nice to have."

---

## 9. Generalisation Strategy

**Decision: Train on a distribution of scenarios, not a single fixed condition**

```python
load_multiplier  = Uniform(0.8, 1.4)   # scale base load up/down — NOT IMPLEMENTED
price_scenario   = Choice(["low", "medium", "high"])  # NOT IMPLEMENTED
ev_penetration   = fixed per experiment (20% / 40% / 60% of households have EVs)  # NOT IMPLEMENTED
```

**Rationale:** the grid operator/stakeholder wants sensitivity across conditions, not just
performance under one fixed scenario. The EV-penetration sweep (20/40/60%) is intended as the
primary analytical output of the tool.

**Status: not implemented.** No scenario sampling, no penetration sweep harness exists yet.
Currently every training episode is identical.

---

## 10. Algorithm

**Decision: IPPO with a shared policy** (revised from original DQN decision)

Originally planned as independent DQN per agent (Discrete(3) action space, simple to debug).
Implementation uses **IPPO (PPO per agent, shared `household_policy` weights)** via Ray RLlib
instead — chosen for training efficiency (one network instead of N, faster convergence) and
because RLlib's multi-agent IPPO support made this the more practical path to a working pipeline.

See §5 above for what shared weights changes about the experiment's meaning.

Future phases may still explore:
- Independent (non-shared) learners, to directly test the sensitivity flagged in §5
- QMIX or VDN for genuinely cooperative variants
- Continuous action space with SAC

**Status: implemented** (`rl_engine/ippo_config.py`, `rl_engine/trainer.py`).

---

## 11. Baseline & Evaluation

**Decision (added — not in original doc): RL claims require a baseline comparison**

To support any claim that "RL reduces curtailment" or "smarter selfish behaviour avoids
synchronisation," results must be compared against fixed, non-learned policies run through
identical episodes:

1. **Naive/uncontrolled** — charge at FULL immediately whenever plugged in. Represents today's
   behaviour with flat pricing.
2. **Naive price-following** — charge at the cheapest available price slot, no learning. Represents
   the future-electrification risk: many customers reacting identically to the same dynamic price
   signal, without any grid-awareness — the synchronisation failure mode the Core Thesis is
   concerned with.
3. **RL (shared-policy IPPO)** — the candidate that should ideally beat #2's synchronisation problem
   while remaining purely selfish.

Metrics per policy, evaluated over held-out episode seeds (not used in training): §14a curtailment
event count, transformer/line loading distribution, SoC target satisfaction rate, total cost.

**Status: not implemented.** No baseline policy, no evaluation harness exists yet (`metrics.py` is
still a stub). This is the highest-priority gap — without it, no quantitative claim about RL vs.
naive behaviour can be made.

---

## 12. Value for Grid Operators / Stakeholders

The tool aims to answer:

> "If customers become price-reactive (due to future dynamic tariffs and rising EV adoption), does
> §14a's curtailment mechanism cope — or does naive price-following create new synchronised-spike
> problems? Does smarter (RL) selfish charging behaviour make this better or worse?"

Key outputs once §9 and §11 are implemented:
- Number of §14a interventions per day, per EV penetration level, per policy (naive / naive
  price-following / RL)
- Transformer and line loading statistics
- SoC satisfaction rate per policy
- Whether RL avoids or worsens synchronised-spike behaviour relative to naive price-following

**Important framing for any audience:** results characterise a *mechanism* (is this approach
workable, what does it predict directionally) under named simplifications — not a forecast of real
grid behaviour for any specific network. The grid model is an OSM-derived approximation; real
deployment would require a DSO's actual topology and validated load/price data.
