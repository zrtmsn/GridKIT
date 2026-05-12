# core/config.py
# ─────────────────────────────────────────────────────────────
# Runtime configuration via environment variables or .env file.
# Usage:  from core.config import settings
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
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

    # ── Dashboard ────────────────────────────────────────────
    dashboard_port: int = 8501
    dashboard_host: str = "localhost"


# Singleton — import this everywhere
settings = Settings()