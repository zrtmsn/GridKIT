# core/constants.py
# ─────────────────────────────────────────────────────────────
# Project-wide constants for GridKIT.
# Import ONLY from this file — never hardcode values in modules.
# ─────────────────────────────────────────────────────────────

# ── §14a EnWG ────────────────────────────────────────────────
MAX_CONTROLLED_POWER_KW: float = 4.2      # hard cap per controllable device
GRID_CONNECTION_KW: float = 11.0          # standard household grid connection

# ── EV charging levels ───────────────────────────────────────
EV_POWER_OFF_KW: float = 0.0
EV_POWER_HALF_KW: float = 3.7
EV_POWER_FULL_KW: float = 7.4

# ── Episode / time ───────────────────────────────────────────
TIMESTEP_MINUTES: int = 15
TIMESTEP_HOURS: float = TIMESTEP_MINUTES / 60.0   # 0.25 h
EPISODE_STEPS: int = 96                            # 24 h / 15 min

# ── EV scenario distributions (mean, std) ───────────────────
EV_ARRIVAL_HOUR_MEAN: float = 18.0
EV_ARRIVAL_HOUR_STD: float = 1.0
EV_DEPARTURE_HOUR_MEAN: float = 7.0
EV_DEPARTURE_HOUR_STD: float = 0.5
EV_INITIAL_SOC_MEAN: float = 0.30
EV_INITIAL_SOC_STD: float = 0.10
EV_TARGET_SOC: float = 0.80           # fixed target for all agents
EV_BATTERY_CAPACITY_KWH: float = 77.0 # aligns with GridCreator (main_functions.py:615)

# ── Load / price scenario ────────────────────────────────────
LOAD_MULTIPLIER_MIN: float = 0.8   # scale factor applied to BDEW H0 base load each episode
LOAD_MULTIPLIER_MAX: float = 1.4   # sampled from Uniform(MIN, MAX) to vary grid stress across episodes

# ── Reward weights ───────────────────────────────────────────
REWARD_ELECTRICITY_COST_WEIGHT: float = 0.1   # small per-timestep penalty: weight * kWh delivered
REWARD_SOC_COMPLETION_BONUS: float = 10.0     # terminal bonus when SoC ≥ target at departure
REWARD_SOC_MISS_PENALTY: float = -10.0        # terminal penalty when SoC < target at departure

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
IPPO_NUM_ENV_RUNNERS: int = 4      # number of parallel environment workers for data collection
IPPO_NUM_EVALUATION_ENV_RUNNERS: int = 0  # number of parallel environment workers for evaluation
IPPO_EVALUATION_INTERVAL: int = 5  # run evaluation every N training iterations
IPPO_NUM_GPUS: int = 0             # will be updated soon
IPPO_NUM_CPUS: int = 0             # will be updated soon

# ── Observation / action dims ────────────────────────────────
OBS_DIM: int = 5
ACTION_DIM: int = 3   # OFF=0, HALF=1, FULL=2

# ── Grid / network ───────────────────────────────────────────
NOMINAL_VOLTAGE_KV: float = 0.4        # low-voltage distribution
TRANSFORMER_OVERLOAD_THRESHOLD: float = 1.0   # p.u. — above = curtailment
LINE_OVERLOAD_THRESHOLD: float = 1.0          # p.u.

# ── EV penetration levels (experiment axis) ──────────────────
EV_PENETRATION_LEVELS: tuple[float, ...] = (0.20, 0.40, 0.60)

# ── Action → power mapping ───────────────────────────────────
ACTION_TO_KW: dict[int, float] = {0: EV_POWER_OFF_KW, 1: EV_POWER_HALF_KW, 2: EV_POWER_FULL_KW}

# ── RLlib / Ray ──────────────────────────────────────────────
RLLIB_ENV_REGISTRY_NAME: str = "GridEnv-v0"    # registered name for Gymnasium/RLlib — do not change without updating the registration hook
RLLIB_DEFAULT_NUM_ITERATIONS: int = 3          # low default for fast iteration on consumer hardware; increase for real training runs
