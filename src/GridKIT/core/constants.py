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
EV_BATTERY_CAPACITY_KWH: float = 60.0 # default battery size

# ── Load / price scenario ────────────────────────────────────
LOAD_MULTIPLIER_MIN: float = 0.8
LOAD_MULTIPLIER_MAX: float = 1.4

# ── Reward weights ───────────────────────────────────────────
REWARD_ELECTRICITY_COST_WEIGHT: float = 0.1
REWARD_SOC_COMPLETION_BONUS: float = 10.0
REWARD_SOC_MISS_PENALTY: float = -10.0

# ── DQN hyperparameters (defaults) ──────────────────────────
DQN_HIDDEN_SIZE: int = 64
DQN_LEARNING_RATE: float = 1e-3
DQN_GAMMA: float = 0.99
DQN_EPSILON_START: float = 1.0
DQN_EPSILON_END: float = 0.05
DQN_EPSILON_DECAY: float = 0.995
DQN_TARGET_UPDATE_STEPS: int = 100
DQN_BATCH_SIZE: int = 64
DQN_REPLAY_BUFFER_SIZE: int = 10_000

# ── Observation / action dims ────────────────────────────────
OBS_DIM: int = 5
ACTION_DIM: int = 3   # OFF=0, HALF=1, FULL=2

# ── Grid / network ───────────────────────────────────────────
NOMINAL_VOLTAGE_KV: float = 0.4        # low-voltage distribution
TRANSFORMER_OVERLOAD_THRESHOLD: float = 1.0   # p.u. — above = curtailment
LINE_OVERLOAD_THRESHOLD: float = 1.0          # p.u.

# ── EV penetration levels (experiment axis) ──────────────────
EV_PENETRATION_LEVELS: tuple[float, ...] = (0.20, 0.40, 0.60)

# ── Action → power mapping ───────────────────────────────
ACTION_TO_KW: dict[int, float] = {0: EV_POWER_OFF_KW, 1: EV_POWER_HALF_KW, 2: EV_POWER_FULL_KW}