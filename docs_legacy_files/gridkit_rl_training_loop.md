# GridKIT — RL Training Loop

> Technical reference for the multi-agent reinforcement learning training loop. Last updated: June 2026.
> Supersedes the earlier DQN/independent-learner description — see `gridkit_rl_decisions.md` §5 and §10 for why the algorithm changed.

---

## Overview

GridKIT uses **IPPO (Independent PPO) with a single shared policy** (`household_policy`) across all
household agents, trained via Ray RLlib. Agents act independently at runtime — no communication,
no shared reward — but they all execute the same learned policy network. The grid environment
physically couples them through shared network infrastructure: power flow is computed every
timestep to enforce §14a curtailment.

See `gridkit_rl_decisions.md` (Core Thesis) for why this matters: shared weights mean the
experiment models "every household adopts the same charging strategy," not "every household
learns its own independent quirks" — a meaningful departure from the original design intent that
should be stated explicitly whenever this loop is presented.

---

## Class Architecture

### `GridEnv` (`grid_model/environment.py`)

Implements `GridEnvProtocol`. Wraps the physical simulation of the low-voltage network.

**Key responsibilities:**
- `reset()` — initialises EV states, returns initial observations for all agents
- `step(actions: dict)` — accepts one action per agent, runs the full power flow, applies §14a
  curtailment if needed, updates SoC for each EV, returns observations, rewards, done flags

**What happens inside `step()`:**
```
1. Collect requested power levels from all agents
2. Add base loads (currently flat constant — BDEW H0 profile not yet wired in)
3. Run PyPSA power flow over the network
4. Check transformer and line loading against rated capacity
5. Apply §14a curtailment if overload is detected
6. Update SoC for each EV based on actual (curtailed) power delivered
7. Compute rewards
8. Return {agent_id: (next_obs, reward, done)}
```

> **Why power flow must run every step:** curtailment depends on the *aggregate* load across all
> agents, not any single household. One agent requesting FULL (7.4 kW) may be safe in isolation;
> ten agents simultaneously requesting FULL can overload the transformer. Power flow is the only
> way to detect this emergent overload.

### `GridEnvRLlibWrapper` (`rl_engine/grid_env_rllib_wrapper.py`)

Dependency-injection wrapper: takes any `GridEnvProtocol` instance and translates between core
models (`Observation`, `StepResult`, `ChargingAction`) and the numpy/dict format Ray RLlib's
multi-agent API expects.

### `household_policy` (`rl_engine/ippo_config.py`)

A single PPO policy shared by every agent. `create_ippo_config()` returns a `PPOConfig` that maps
all agent IDs to this one policy. There is no per-household network — this is the architectural
choice flagged in the Core Thesis.

### `Trainer` (`rl_engine/trainer.py`)

Outer orchestration class. Accepts an `env_factory` and `config_func`, registers the environment
with Ray, runs the RLlib training loop, fires progress callbacks (`callbacks.py`).

---

## Observation Vector

Each agent receives a 5-dimensional local observation — only what a smart meter at its own
household can measure:

| Index | Feature | Notes |
|---|---|---|
| 0 | `current_soc / target_soc` | 0–1 progress toward goal |
| 1 | `time_until_departure / episode_length` | 0–1 urgency signal |
| 2 | `current_electricity_price` | **raw, unnormalised tariff value** — see known issue below |
| 3 | `base_load_kw` | inflexible household consumption — currently a flat constant |
| 4 | `outdoor_temperature_c` | context for future heat pump phase — not yet populated from a real source |

Agents do **not** observe: neighbour actions, transformer load, line utilisation, or whether other
agents are being curtailed.

**Known issue — price normalisation:** because price is fed raw, the policy's notion of "cheap" is
entirely relative to whatever range it saw in training; it has no built-in sense of scale. Once
dynamic pricing is wired in, consider normalising against the known day-ahead price curve
(percentile or min-max) rather than feeding raw €/kWh.

---

## Action Space

```python
class ChargingAction(IntEnum):
    OFF  = 0  # 0.0 kW
    HALF = 1  # 3.7 kW
    FULL = 2  # 7.4 kW
```

Gymnasium-style multi-agent interface via RLlib, `spaces.Discrete(3)` per agent.

---

## Reward Structure

```
r_t = + progress_toward_soc_target            # positive signal each timestep
    - electricity_cost_weight * kWh_delivered # small cost signal each timestep

r_terminal = + large_bonus   if SoC ≥ target at departure
             - large_penalty if SoC <  target at departure
```

§14a curtailment is **not** a separate penalty term. Curtailment reduces delivered energy, which
reduces SoC progress, which reduces reward — an indirect channel, by design (see Core Thesis: the
agent is never told to protect the grid).

---

## Episode Structure

| Parameter | Design target | Current status |
|---|---|---|
| Timestep | 15 minutes | implemented |
| Episode length | 96 steps (24 hours) | implemented |
| EV arrival time | `N(18:00, 1h)` | **not implemented** — hardcoded `arrival_step=0` |
| Initial SoC | `N(0.30, 0.10)` | **not implemented** — fixed constant |
| Departure time | `N(07:00, 0.5h)` | **not implemented** — hardcoded `departure_step=95` |
| Target SoC | 0.80 (fixed) | implemented |
| Load profile | BDEW H0 standard profile | **not implemented** — flat constant |
| Price profile | ENTSO-E day-ahead prices | **not implemented** — flat constant |

Every episode is currently identical. No scenario sampling exists yet — this is the biggest gap
between the original design and what can be demonstrated today.

---

## Training Loop (as implemented via Ray RLlib)

```
Trainer.run()
│
├── env_factory() registered with Ray as a multi-agent env (wraps GridEnv via GridEnvRLlibWrapper)
├── config_func() builds the PPOConfig with shared household_policy
│
└── Ray RLlib internally:
      for iteration in range(num_iterations):
          │
          ├── rollout workers run episodes against the env
          │     (each of the 96 steps: actions from household_policy → env.step() →
          │      power flow → §14a curtailment if needed → reward → next obs)
          │
          ├── PPO update on collected rollout batches → household_policy weights
          │
          └── DefaultCallback.on_iteration_end(iteration, mean_reward, mean_len)
                # prints progress; no EpisodeMetrics/SimResult populated yet
```

This is intentionally less detailed than the previous DQN-era documentation (which specified
replay buffers, epsilon-greedy, target networks) because those mechanics are now handled inside
Ray RLlib's PPO implementation rather than hand-rolled — the project-specific logic lives in the
environment, the reward, the observation, and the IPPO config, not in a custom training loop.

---

## Generalisation Strategy — not yet implemented

```python
load_multiplier = Uniform(0.8, 1.4)          # NOT IMPLEMENTED
price_scenario  = Choice(["low", "medium", "high"])  # NOT IMPLEMENTED
ev_penetration  = fixed per experiment (20% / 40% / 60%)  # NOT IMPLEMENTED
```

No scenario sampling and no penetration sweep harness exist. This is required before the
EV-penetration sweep (the design doc's "primary analytical output") can be produced.

---

## Baseline & Evaluation — not yet implemented

No naive/uncontrolled or naive-price-following baseline policy exists, and no evaluation harness
runs held-out episodes to measure curtailment count, loading, or SoC satisfaction rate per policy.
Without this, "RL reduces curtailment" cannot currently be quantified. See
`gridkit_rl_decisions.md` §11 for the three-policy comparison plan needed to support that claim.

---

## Future Extensions

| Feature | Change required |
|---|---|
| Independent (non-shared) learners | Separate policy per agent in `ippo_config.py`; directly tests the sensitivity flagged in the Core Thesis |
| QMIX / cooperative variants | Replace IPPO with a joint value function approach |
| Continuous action space | Replace `Discrete(3)` with `Box`, swap PPO discrete head for continuous |
| Heat pumps / battery storage | Add device state variables; extend the 5-dim observation vector |
| Partial observability relaxed | Add transformer load or aggregate neighbourhood load to observation |
