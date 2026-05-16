# GridKIT — RL Training Loop

> Technical reference for the multi-agent reinforcement learning training loop. Last updated: May 2026.

---

## Overview

GridKIT uses independent Deep Q-Networks (DQN) for each household agent. Agents do not share parameters or communicate. The grid environment physically couples them through shared network infrastructure — power flow is computed on every timestep to enforce §14a curtailment rules.

---

## Class Architecture

### `GridKITEnv(gymnasium.Env)`

The central environment. Wraps the physical simulation of the low-voltage distribution network (OSM topology of Karlsruhe).

**Key responsibilities:**
- `reset()` — samples a new episode scenario from data distributions, returns initial observations for all agents
- `step(actions: dict)` — accepts one action per agent, runs the full power flow calculation, applies §14a curtailment if needed, updates SoC for each EV, returns observations, rewards, and done flags

**What happens inside `step()`:**

```
1. Collect requested power levels from all agents
2. Add inflexible base loads (BDEW H0 profile)
3. Run power flow over OSM network topology
4. Check transformer and line loading against rated capacity
5. Apply §14a curtailment if overload is detected
6. Update SoC for each EV based on actual (curtailed) power delivered
7. Compute rewards
8. Return {agent_id: (next_obs, reward, done)}
```

> **Why power flow must run every step:** Curtailment depends on the *aggregate* load across all agents — not any single household. One agent requesting FULL (7.4 kW) may be safe in isolation. Ten agents simultaneously requesting FULL will overload the transformer. The power flow calculation is the only way to detect this emergent overload.

---

### `HouseholdAgent`

One instance per household. Operates fully independently — no awareness of neighbours.

**Components:**
- `DQNetwork` — main network (updated every step)
- `DQNetwork` — target network (frozen copy, updated every N steps)
- `ReplayBuffer` — stores past transitions `(s, a, r, s', done)`

**Key methods:**

```python
select_action(obs, epsilon) -> action
    # With probability epsilon: random action from {OFF, HALF, FULL}
    # Otherwise: argmax over Q(obs, ·) from main network

store_transition(s, a, r, s_next, done)
    # Write transition to replay buffer

learn()
    # Sample mini-batch from replay buffer
    # Compute TD target: r + γ · max_a Q_target(s', a)
    # Minimise (Q_main(s, a) - TD_target)²
    # Every N steps: copy main weights → target network
```

---

### `DQNetwork(nn.Module)`

Simple multi-layer perceptron.

| Layer | Size | Notes |
|---|---|---|
| Input | 5 | Observation vector (see below) |
| Hidden 1 | 64 | ReLU |
| Hidden 2 | 64 | ReLU |
| Output | 3 | Q-values for OFF / HALF / FULL |

---

### `ReplayBuffer`

Fixed-size circular buffer. Stores transitions as tuples `(s, a, r, s', done)` and returns random mini-batches for training. Sampling is only enabled once the buffer contains at least `batch_size` transitions.

---

### `Trainer`

Outer orchestration class. Creates the environment and all agents, runs the episode loop, logs metrics.

---

## Observation Vector

Each agent receives a 5-dimensional local observation — only what a smart meter at its own household can measure:

| Index | Feature | Normalisation |
|---|---|---|
| 0 | `current_soc / target_soc` | 0–1 progress toward goal |
| 1 | `time_until_departure / episode_length` | 0–1 urgency signal |
| 2 | `current_electricity_price` | raw tariff value |
| 3 | `base_load_kw` | inflexible household consumption |
| 4 | `outdoor_temperature_c` | environment context |

Agents do **not** observe: neighbour actions, transformer load, line utilisation, or whether other agents are being curtailed.

---

## Action Space

```python
class ChargingAction(IntEnum):
    OFF  = 0  # 0.0 kW
    HALF = 1  # 3.7 kW
    FULL = 2  # 7.4 kW
```

Gymnasium interface: `spaces.Discrete(3)`

---

## Reward Structure

```
r_t = + progress_toward_soc_target          # positive signal each timestep
    - electricity_cost_weight * kWh_delivered # small cost signal each timestep

r_terminal = + large_bonus   if SoC ≥ target at departure
             - large_penalty if SoC <  target at departure
```

§14a curtailment is **not** a separate penalty term. When the grid curtails an agent's charging power, less energy is delivered → less SoC progress → lower cumulative reward. The penalty for missing the departure target is the core motivation for the agent to act proactively rather than waiting.

---

## Episode Structure

| Parameter | Value |
|---|---|
| Timestep | 15 minutes |
| Episode length | 96 steps (24 hours) |
| EV arrival time | `N(18:00, 1h)` |
| Initial SoC | `N(0.30, 0.10)` |
| Departure time | `N(07:00, 0.5h)` |
| Target SoC | 0.80 (fixed) |
| Load profile | BDEW H0 standard profile |
| Price profile | ENTSO-E day-ahead prices |

Each call to `reset()` independently samples a new scenario from these distributions.

---

## Training Loop

```
Trainer.train()
│
├── initialise env, agents, replay buffers
│
└── for episode in range(num_episodes):
      │
      ├── obs_dict = env.reset()
      │     # obs_dict = {agent_id: np.array(5,)}
      │
      ├── for t in range(96):  # 96 timesteps = 24 hours
      │     │
      │     ├── actions = {}
      │     │   for each agent:
      │     │       actions[id] = agent.select_action(obs_dict[id], epsilon)
      │     │
      │     ├── next_obs_dict, rewards, dones = env.step(actions)
      │     │     # inside step():
      │     │     #   power flow computed over full network
      │     │     #   §14a curtailment applied if overload
      │     │     #   SoC updated with actual (curtailed) power
      │     │     #   terminal rewards added at departure timestep
      │     │
      │     ├── for each agent:
      │     │       agent.store_transition(
      │     │           obs_dict[id], actions[id],
      │     │           rewards[id], next_obs_dict[id], dones[id]
      │     │       )
      │     │       agent.learn()   # independent, no coordination
      │     │
      │     ├── obs_dict = next_obs_dict
      │     │
      │     └── if all dones: break
      │
      ├── decay epsilon
      │
      └── log metrics:
            - mean SoC satisfaction rate
            - §14a curtailment events per day
            - transformer peak loading
            - mean episode reward per agent
```

---

## Generalisation Strategy

To prevent overfitting to a single set of conditions, each episode samples from a range of scenarios:

```python
load_multiplier = Uniform(0.8, 1.4)          # scale base load up/down
price_scenario  = Choice(["low", "medium", "high"])
ev_penetration  = fixed per experiment        # 20% / 40% / 60%
```

Experiments are run across EV penetration levels (20% → 40% → 60%) to produce the primary analytical output: how curtailment frequency scales with EV adoption under selfish RL charging behaviour.

---

## Key Design Principles

**Independent learners.** Each agent runs its own DQN with its own replay buffer. No parameter sharing in Phase 1. This is consistent with the selfish, non-cooperative agent model and reflects the real-world structure of §14a.

**Implicit coordination through physics.** Agents do not communicate, but they are coupled through the power flow calculation inside `env.step()`. When many agents charge simultaneously, the transformer overloads, curtailment is applied, and all affected agents receive lower SoC progress — a distributed signal that emergent coordination may improve individual outcomes.

**Curtailment as environment physics, not reward shaping.** §14a curtailment is not a penalty term. It is a physical constraint enforced by the environment. The agent experiences it indirectly through reduced energy delivery. This ensures the reward signal remains aligned with the household's actual goal: charge the EV to target SoC before departure.

---

## Future Extensions (Post Phase 1)

| Feature | Change required |
|---|---|
| Parameter sharing across agents | Share single DQN weights across all `HouseholdAgent` instances |
| QMIX / cooperative variants | Replace independent learner with joint value function |
| Continuous action space | Replace DQN with SAC; change `spaces.Discrete(3)` to `spaces.Box` |
| Heat pumps / battery storage | Add device state variables to `HouseholdAgent`; extend observation vector |
| Partial observability | Add transformer load or aggregate neighbourhood load to observation |
