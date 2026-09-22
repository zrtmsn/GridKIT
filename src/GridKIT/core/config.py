# core/config.py
# ─────────────────────────────────────────────────────────────
# Runtime configuration via environment variables or .env file.
# Usage:  from core.config import settings
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.constants import (
    DQN_BATCH_SIZE,
    DQN_EPSILON_DECAY,
    DQN_EPSILON_END,
    DQN_EPSILON_START,
    DQN_GAMMA,
    DQN_HIDDEN_SIZE,
    DQN_LEARNING_RATE,
    DQN_REPLAY_BUFFER_SIZE,
    DQN_TARGET_UPDATE_STEPS,
    EV_PENETRATION_LEVELS,
    IPPO_BATCH_SCALING_REF_AGENTS,
    IPPO_CLIP_EPS,
    IPPO_EARLY_STOP_MIN_DELTA,
    IPPO_ENTROPY_COEFF,
    IPPO_EVALUATION_INTERVAL,
    IPPO_GAE_LAMBDA,
    IPPO_GAMMA,
    IPPO_HIDDEN_SIZE,
    IPPO_LEARNING_RATE,
    IPPO_MAX_GRAD_NORM,
    IPPO_MINIBATCH_SIZE,
    IPPO_N_EPOCHS,
    IPPO_NUM_CPUS,
    IPPO_NUM_ENV_RUNNERS,
    IPPO_NUM_EVALUATION_ENV_RUNNERS,
    IPPO_NUM_GPUS,
    IPPO_ROLLOUT_STEPS,
    IPPO_TRAIN_BATCH_SIZE,
    IPPO_VALUE_COEFF,
    PIPELINE_EARLY_STOP_PATIENCE,
    PIPELINE_EARLY_STOP_SMOOTH_WINDOW,
    PIPELINE_MIN_TRAINING_ITERATIONS,
    RLLIB_DEFAULT_NUM_EPISODES,
    RLLIB_ENV_REGISTRY_NAME,
)


class Settings(BaseSettings):
    """
    All runtime settings for GridKIT.
    Values are loaded from environment variables or a .env file.
    Defaults match the Phase 1 prototype.

    Copy .env.example → .env and adjust as needed.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── General ──────────────────────────────────────────────
    project_name: str = "GridKIT"
    debug: bool = False
    random_seed: int = 42
    # selects which rl_engine backend to use; controls which hyperparameter block is active
    algorithm: Literal["dqn", "ippo"] = "dqn"

    # ── Paths ────────────────────────────────────────────────
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")
    scenario_csv_dir: Path = Path("data/scenarios")

    # ── Episode / simulation ─────────────────────────────────
    num_episodes: int = 1000
    ev_penetration: float = Field(
        default=0.20,
        description="Fraction of households with an EV. One of 0.20 / 0.40 / 0.60",
    )

    @field_validator("ev_penetration")
    @classmethod
    def ev_penetration_must_be_valid(cls, v: float) -> float:
        if v not in EV_PENETRATION_LEVELS:
            raise ValueError(f"ev_penetration must be one of {EV_PENETRATION_LEVELS}, got {v}")
        return v

    # ── Stub network (Phase 1) ───────────────────────────────
    stub_network_path: Path = Path("data/stub_network.json")

    # ── DQN hyperparameters ──────────────────────────────────
    dqn_hidden_size: int = DQN_HIDDEN_SIZE
    dqn_learning_rate: float = DQN_LEARNING_RATE
    dqn_gamma: float = DQN_GAMMA
    dqn_epsilon_start: float = DQN_EPSILON_START
    dqn_epsilon_end: float = DQN_EPSILON_END
    dqn_epsilon_decay: float = DQN_EPSILON_DECAY
    dqn_target_update_steps: int = DQN_TARGET_UPDATE_STEPS
    dqn_batch_size: int = DQN_BATCH_SIZE
    dqn_replay_buffer_size: int = DQN_REPLAY_BUFFER_SIZE

    # ── IPPO hyperparameters ─────────────────────────────────
    ippo_hidden_size: int = IPPO_HIDDEN_SIZE
    ippo_learning_rate: float = IPPO_LEARNING_RATE
    ippo_gamma: float = IPPO_GAMMA
    ippo_gae_lambda: float = IPPO_GAE_LAMBDA
    ippo_clip_eps: float = IPPO_CLIP_EPS
    ippo_n_epochs: int = IPPO_N_EPOCHS
    ippo_rollout_steps: int = IPPO_ROLLOUT_STEPS
    ippo_minibatch_size: int = IPPO_MINIBATCH_SIZE
    ippo_train_batch_size: int = IPPO_TRAIN_BATCH_SIZE
    ippo_entropy_coeff: float = IPPO_ENTROPY_COEFF
    ippo_value_coeff: float = IPPO_VALUE_COEFF
    ippo_max_grad_norm: float = IPPO_MAX_GRAD_NORM
    ippo_num_env_runners: int = IPPO_NUM_ENV_RUNNERS
    ippo_num_evaluation_env_runners: int = IPPO_NUM_EVALUATION_ENV_RUNNERS
    ippo_evaluation_interval: int = IPPO_EVALUATION_INTERVAL
    ippo_num_gpus: int = IPPO_NUM_GPUS
    ippo_num_cpus: int = IPPO_NUM_CPUS
    # Agent count of the reference grid that IPPO_TRAIN_BATCH_SIZE was tuned
    # for; Trainer scales batch sizes with ceil(n_agents / this) so the
    # transitions PER AGENT per update stay constant across grid sizes.
    ippo_batch_scaling_ref_agents: int = IPPO_BATCH_SCALING_REF_AGENTS
    # Plateau threshold: minimum improvement of episode_return_mean that still
    # counts as progress for convergence-based early stopping.
    ippo_early_stop_min_delta: float = IPPO_EARLY_STOP_MIN_DELTA

    # ── RLlib / Ray ───────────────────────────────────────────
    rllib_env_registry_name: str = RLLIB_ENV_REGISTRY_NAME
    rllib_default_num_episodes: int = RLLIB_DEFAULT_NUM_EPISODES

    # ── Convergence-based early stopping ─────────────────────
    # Defaults for Trainer.run(); the hard ceiling for UI-launched runs
    # (PIPELINE_MAX_TRAINING_ITERATIONS) stays a constant in core.constants.
    pipeline_min_training_iterations: int = PIPELINE_MIN_TRAINING_ITERATIONS
    pipeline_early_stop_patience: int = PIPELINE_EARLY_STOP_PATIENCE
    # Sliding window `k` for convergence smoothing: the plateau logic compares
    # the mean of the last k per-iteration reward means (window slides; not the
    # mean since training start, not re-averaged). 1 disables smoothing.
    pipeline_early_stop_smooth_window: int = PIPELINE_EARLY_STOP_SMOOTH_WINDOW

    # ── Dashboard ────────────────────────────────────────────
    dashboard_port: int = 8501
    dashboard_host: str = "localhost"


# Singleton — import this everywhere
settings = Settings()