# core/constants.py
# ─────────────────────────────────────────────────────────────
# Project-wide constants for GridKIT.
# Import ONLY from this file — never hardcode values in modules.
# ─────────────────────────────────────────────────────────────

# ── §14a EnWG ────────────────────────────────────────────────
MIN_GUARANTEED_POWER_KW: float = 4.2      # §14a Mindestleistung — curtailment may not dim below this
MAX_CONTROLLED_POWER_KW: float = MIN_GUARANTEED_POWER_KW  # backward-compat alias (old name was misleading)
GRID_CONNECTION_KW: float = 11.0          # standard household grid connection (capacity, NOT consumption)

# ── EV charging levels ───────────────────────────────────────
EV_POWER_OFF_KW: float = 0.0
EV_POWER_HALF_KW: float = 3.7
EV_POWER_FULL_KW: float = 7.4

# ── Episode / time ───────────────────────────────────────────
TIMESTEP_MINUTES: int = 15
TIMESTEP_HOURS: float = TIMESTEP_MINUTES / 60.0   # 0.25 h
EPISODE_STEPS: int = 96                            # 24 h / 15 min
EPISODE_START_HOUR: int = 12                       # episode starts at noon so the overnight
                                                   # connected window (evening→morning) never wraps

# ── EV scenario distributions (mean, std) ───────────────────
EV_ARRIVAL_HOUR_MEAN: float = 18.0
EV_ARRIVAL_HOUR_STD: float = 1.5    # commuter home-arrival spread (~15:00–21:00)
EV_DEPARTURE_HOUR_MEAN: float = 7.0
EV_DEPARTURE_HOUR_STD: float = 0.5
# realistic daily depletion: ~0.25 SoC ≈ 19 kWh ≈ 110 km/day on a 77 kWh battery.
# (The old 0.30 implied ~38 kWh/night ≈ 200 km/day, which forced every car into a
#  ~5 h full charge and made the flat/immediate baseline overload all evening.)
EV_INITIAL_SOC_MEAN: float = 0.55
EV_INITIAL_SOC_STD: float = 0.10
EV_TARGET_SOC: float = 0.80           # fixed target for all agents
EV_BATTERY_CAPACITY_KWH: float = 77.0 # aligns with GridCreator (main_functions.py:615)

# Per-agent connection-window sampling bounds (in step indices, episode-local).
# With EPISODE_START_HOUR=12: arrival ~18:00 → step 24, departure ~07:00 → step 76.
EV_ARRIVAL_STEP_MIN: int = 1          # earliest sampled arrival
EV_DEPARTURE_STEP_MAX: int = 95       # latest sampled departure (must stay < EPISODE_STEPS)
EV_MIN_CONNECTED_STEPS: int = 8       # enforce departure ≥ arrival + this (≥ 2 h plugged in)

# ── Base load (BDEW H0-like) ─────────────────────────────────
# NOTE: the old code used GRID_CONNECTION_KW (11 kW, the *connection capacity*) as base load,
# which alone saturates the feeder. These are realistic *consumption* values.
HOUSEHOLD_BASE_LOAD_MEAN_KW: float = 0.5   # daily-average household draw
HOUSEHOLD_BASE_LOAD_PEAK_KW: float = 1.5   # evening peak of the H0 shape
LOAD_MULTIPLIER_MIN: float = 0.8   # scale factor applied to BDEW H0 base load each episode
LOAD_MULTIPLIER_MAX: float = 1.4   # sampled from Uniform(MIN, MAX) to vary grid stress across episodes

# ── Dynamic price (day-ahead-like) ───────────────────────────
PRICE_BASE_EUR_KWH: float = 0.30           # mean price level (MEDIUM scenario)
PRICE_PEAK_AMPLITUDE_EUR_KWH: float = 0.15 # depth of overnight trough / height of evening peak
PRICE_NOISE_STD_EUR_KWH: float = 0.02      # per-episode i.i.d. noise on the published curve
PRICE_SCENARIO_MULTIPLIER: dict[str, float] = {"low": 0.7, "medium": 1.0, "high": 1.4}

# ── Scenario 2 (naive price-follow) behaviour ────────────────
NAIVE_PRICE_JITTER_STEPS_STD: float = 4.0  # σ (in steps, 1 h) of start-time jitter; runner sweeps it
                                           # σ→0 = automated/app-driven, large σ = manual/human

# ── Reward weights ───────────────────────────────────────────
# Everything is denominated in EUROS so the trade-offs are legible and defensible:
# one unit of reward = one euro to the household. Energy is priced at what it costs;
# service failures at what a household would plausibly pay to avoid them.
#
# The previous weighting (cost 0.1, SoC ±10) made the SoC outcome worth ~13x the
# ENTIRE daily bill: the best possible price optimisation saved €3/day = 0.3 reward,
# while missing target cost 20. Price was therefore irrelevant to the optimum, and the
# trained agent correctly collapsed onto "charge immediately, ignore price" —
# reproducing scenario 1 and costing MORE than every baseline. Fixing the balance is
# what makes price-responsive behaviour learnable at all.
REWARD_ELECTRICITY_COST_WEIGHT: float = 1.0   # reward in €: weight * bill (import − feed-in)
REWARD_SOC_COMPLETION_BONUS: float = 0.0      # deprecated: the bonus/penalty CLIFF at exactly
                                              # target made a car at 0.799 score like one at 0.2,
                                              # distorting both the policy and the reported metric
REWARD_SOC_MISS_PENALTY: float = -400.0       # €-equivalent of arriving with an EMPTY battery,
                                              # applied to the shortfall SQUARED (see below)
# Why quadratic, and why so large a coefficient. A LINEAR penalty prices the first 1% of
# shortfall the same as the last, which does not match how a driver experiences it: 2%
# short is irrelevant (range margin absorbs it), 40% short means missing the commute. A
# linear −25 was measured to be exploitable — the agent undercharged by 13.9%, saving
# €4.57/day against a €3.48 penalty, which was rational under that reward and left cars
# at 0.69 SoC. Squaring makes small deviations nearly free (so price still steers the
# policy) while deep undercharging becomes prohibitive:
#     2% → €0.16    5% → €1.00    14% → €7.84    100% → €400
# Calibration: the agent gains ≈€33 per unit of shortfall on this grid, so the optimum
# sits at s* = 33/(2K) ≈ 4% — cars arriving near 0.77, still load-shifting for price.

# ── DQN hyperparameters (defaults) ──────────────────────────
DQN_HIDDEN_SIZE: int = 64          # neurons per hidden layer in the Q-network
DQN_LEARNING_RATE: float = 1e-3
DQN_GAMMA: float = 0.99            # discount factor — how much future rewards are valued (0=myopic, 1=far-sighted)
DQN_EPSILON_START: float = 1.0     # initial exploration rate — agent acts randomly at the start
DQN_EPSILON_END: float = 0.05      # minimum exploration rate — always keeps 5% random actions
DQN_EPSILON_DECAY: float = 0.995   # multiplicative decay applied to epsilon after each episode
DQN_TARGET_UPDATE_STEPS: int = 100 # copy main network weights → target network every N steps
DQN_BATCH_SIZE: int = 64           # number of transitions sampled from replay buffer per learning step
DQN_REPLAY_BUFFER_SIZE: int = 10_000  # max transitions stored; oldest are overwritten when full

# ── IPPO hyperparameters (defaults) ─────────────────────────
IPPO_HIDDEN_SIZE: int = 64         # neurons per hidden layer in actor and critic networks
IPPO_LEARNING_RATE: float = 3e-4
IPPO_GAMMA: float = 0.99           # discount factor
IPPO_GAE_LAMBDA: float = 0.95      # GAE smoothing: 0=pure TD, 1=pure MC
IPPO_CLIP_EPS: float = 0.2         # PPO clipping range for the probability ratio
IPPO_N_EPOCHS: int = 10            # gradient update passes over one collected rollout
IPPO_ROLLOUT_STEPS: int = 96       # steps collected per agent before each update (= 1 episode)
IPPO_MINIBATCH_SIZE: int = 64      # minibatch size for SGD updates within one PPO epoch
IPPO_TRAIN_BATCH_SIZE: int = 512   # total steps collected per update cycle across all workers
IPPO_ENTROPY_COEFF: float = 0.01   # entropy bonus weight — encourages exploration
IPPO_VALUE_COEFF: float = 0.5      # critic loss weight relative to actor loss
IPPO_MAX_GRAD_NORM: float = 0.5    # gradient clipping threshold
IPPO_NUM_ENV_RUNNERS: int = 2      # parallel environment workers for data collection
IPPO_NUM_EVALUATION_ENV_RUNNERS: int = 1  # parallel environment workers for evaluation
IPPO_EVALUATION_INTERVAL: int = 5  # run evaluation every N training iterations
IPPO_NUM_GPUS: int = 0
IPPO_NUM_CPUS: int = 0

# ── Observation / action dims ────────────────────────────────
OBS_DIM: int = 7   # [soc_progress, time_urgency, price, base_load, temp, local_voltage, recent_curtailment]
ACTION_DIM: int = 3   # OFF=0, HALF=1, FULL=2

# ── Observation normalization (raw → [0,1], applied in the RLlib wrapper) ──
PRICE_NORM_MAX_EUR_KWH: float = 0.60   # price / this
BASE_LOAD_NORM_MAX_KW: float = 3.0     # base_load_kw / this
TEMP_MIN_C: float = -10.0              # (temp - MIN) / (MAX - MIN)
TEMP_MAX_C: float = 40.0
VOLTAGE_MIN_PU: float = 0.90           # (v - MIN) / (MAX - MIN); LV undervoltage limit ≈ 0.9 pu
VOLTAGE_MAX_PU: float = 1.10

# ── Grid / network ───────────────────────────────────────────
NOMINAL_VOLTAGE_KV: float = 0.4        # low-voltage distribution
TRANSFORMER_OVERLOAD_THRESHOLD: float = 1.0   # p.u. — above = curtailment
LINE_OVERLOAD_THRESHOLD: float = 1.0          # p.u.
# Peak loading is RECORDED from here up, so the overload map can shade cables that ran
# hot without tripping. A village has thousands of lines and most carry almost nothing;
# keeping only the loaded ones is what makes the per-run record small enough to store.
LINE_WATCH_THRESHOLD: float = 0.5             # p.u.
TRANSFORMER_REACTANCE_PU: float = 0.04        # ~4% short-circuit reactance (needed for a solvable pf)
TRANSFORMER_RESISTANCE_PU: float = 0.01

# ── EV penetration levels (experiment axis) ──────────────────
EV_PENETRATION_LEVELS: tuple[float, ...] = (0.20, 0.40, 0.60)

# ── Action → power mapping ───────────────────────────────────
ACTION_TO_KW: dict[int, float] = {0: EV_POWER_OFF_KW, 1: EV_POWER_HALF_KW, 2: EV_POWER_FULL_KW}

# ═════════════════════════════════════════════════════════════
# MULTI-DEVICE HOUSEHOLD (EV + battery + heat pump + PV)
# Each device at a household is its own RL agent; one shared policy per
# device *type*. Real exogenous profiles come from grid_model/device_profiles.py
# (GridCreator/pyCity). Device dynamics live in grid_model/environment.py; these
# are the parameters + action encodings that core exposes to every module.
# ═════════════════════════════════════════════════════════════

# ── Device types (agent-id suffix → shared policy) ───────────
DEVICE_EV: str = "ev"
DEVICE_BATTERY: str = "battery"
DEVICE_HEAT_PUMP: str = "hp"
DEVICE_PV: str = "pv"
DEVICE_TYPES: tuple[str, ...] = (DEVICE_EV, DEVICE_BATTERY, DEVICE_HEAT_PUMP, DEVICE_PV)
# PV is folded into the battery decision: it is EXOGENOUS generation (always
# self-consumed via the meter, surplus exported at the feed-in tariff), NOT an
# agent — an independent sell/self-consume PV action is degenerate when
# feed-in < retail. Controllable = one RL agent each; one shared policy per type.
CONTROLLABLE_DEVICE_TYPES: tuple[str, ...] = (DEVICE_EV, DEVICE_BATTERY, DEVICE_HEAT_PUMP)

# ── PV / feed-in ─────────────────────────────────────────────
# Feed-in must pay LESS than retail price, otherwise "sell everything" always
# wins and the self-consume/sell action is meaningless. (EEG feed-in ≈ 0.08 €/kWh
# vs. retail ≈ 0.30 €/kWh — self-consumption is worth ~0.22 €/kWh.)
FEED_IN_TARIFF_EUR_KWH: float = 0.08
PV_PEAK_KWP_MIN: float = 3.0        # per-household PV size sampled from [MIN, MAX]
PV_PEAK_KWP_MAX: float = 10.0

# ── Battery (home storage) ───────────────────────────────────
BATTERY_CAPACITY_KWH: float = 10.0  # typical home battery (e.g. ~1–2 EV-equivalent hours)
BATTERY_MAX_POWER_KW: float = 5.0   # charge/discharge power at CHARGE/DISCHARGE
BATTERY_EFFICIENCY: float = 0.95    # one-way (round-trip ≈ 0.90)
BATTERY_INITIAL_SOC: float = 0.5
BATTERY_MIN_SOC: float = 0.05       # usable-window floor
BATTERY_MAX_SOC: float = 0.95       # usable-window ceiling

# ── Heat pump (thermostatically-controlled flexible load) ────
# Modelled as a thermal buffer (house inertia) with a comfort floor, mirroring
# the EV's SoC target: running the HP charges the buffer, the weather-driven heat
# demand drains it. The agent may pre-heat / coast, but must keep comfort ≥ floor.
HP_RATED_ELECTRIC_KW: float = 3.0        # nameplate electric draw when HEAT
HP_COP: float = 3.0                      # seasonal-average electric→thermal factor
HP_THERMAL_CAPACITY_KWH: float = 8.0     # thermal-buffer size (house inertia)
HP_INITIAL_THERMAL_SOC: float = 0.7      # buffer level at episode start
HP_COMFORT_MIN_SOC: float = 0.30         # comfort floor on the thermal buffer
HP_TEMP_THRESHOLD_C: float = 15.0        # heating demanded below this (GridCreator schedule)

# ── Extra reward terms (household-net reward; see environment._reward) ─
# All four device-agents in a house SHARE one reward = minimize net electricity
# bill + meet EV SoC + meet HP comfort. These are the non-bill terminal terms.
REWARD_HP_COMFORT_BONUS: float = 10.0        # buffer ≥ floor throughout / at end
# Same € scale as the EV term: comfort loss is a service failure, priced like one. Kept
# proportional to the deficit, as it already was — only the magnitude moves, so that
# raising the cost weight to real euros does not silently demote comfort by 10x.
REWARD_HP_COMFORT_MISS_PENALTY: float = -25.0

# ── Per-device action encodings ──────────────────────────────
# Battery: signed power (kW). DISCHARGE(0) supplies the house, CHARGE(2) stores.
BATTERY_ACTION_TO_KW: dict[int, float] = {
    0: -BATTERY_MAX_POWER_KW,   # DISCHARGE
    1: 0.0,                     # IDLE
    2: +BATTERY_MAX_POWER_KW,   # CHARGE
}
# Heat pump: electric draw (kW). OFF(0) / HEAT(1).
HP_ACTION_TO_KW: dict[int, float] = {0: 0.0, 1: HP_RATED_ELECTRIC_KW}
# PV: no action — exogenous generation folded into the battery decision.

# ── Per-device action / observation dims ─────────────────────
ACTION_DIM_BY_DEVICE: dict[str, int] = {
    DEVICE_EV: 3, DEVICE_BATTERY: 3, DEVICE_HEAT_PUMP: 2,
}
# Target unified observation width for the multi-device env (uniform layout across
# device types; each agent's device_soc/time_urgency carry its OWN device state).
# The env rewrite bumps OBS_DIM from 7 → this and repopulates obs_norm. See
# core.models.Observation. NOT yet active while the EV-only path uses OBS_DIM=7.
OBS_DIM_MULTIDEVICE: int = 10

# Normalization bounds for the new observation fields (raw → [0,1]).
NET_LOAD_NORM_MAX_KW: float = 15.0      # household net (consumption − PV) magnitude
PV_GEN_NORM_MAX_KW: float = 10.0        # ≈ PV_PEAK_KWP_MAX

# ── RLlib / Ray ──────────────────────────────────────────────
RLLIB_ENV_REGISTRY_NAME: str = "GridEnv-v0"    # registered name for Gymnasium/RLlib — do not change without updating the registration hook
RLLIB_DEFAULT_NUM_EPISODES: int = 3            # low default for fast iteration on consumer hardware; increase for real training runs

# ── UI-launched training runs (scripts/train_run.py) ──────────
# The map UI never exposes these — a user picks a network + device mix, not
# RL hyperparameters. Fixed here so "launch training" behaves the same for
# every run regardless of who clicks it.
PIPELINE_TRAINING_ITERATIONS: int = 20   # IPPO training iterations per UI-launched run
PIPELINE_EVALUATION_SEEDS: int = 5       # evaluation episodes per scenario after training
