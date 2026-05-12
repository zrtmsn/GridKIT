# core/models.py
# ─────────────────────────────────────────────────────────────
# Single source of truth for all inter-module data structures.
# Modules communicate ONLY through these models — no direct
# cross-module imports allowed.
# ─────────────────────────────────────────────────────────────
 
from __future__ import annotations
 
from enum import IntEnum, StrEnum
from typing import Optional
from pydantic import BaseModel, Field, model_validator
 
from core.constants import (
    EV_BATTERY_CAPACITY_KWH,
    EV_TARGET_SOC,
)
 
 
# ══════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════
 
class ChargingAction(IntEnum):
    OFF  = 0   # 0.0 kW
    HALF = 1   # 3.7 kW
    FULL = 2   # 7.4 kW
 
 
class PriceScenario(StrEnum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
 
 
# ══════════════════════════════════════════════════════════════
# Grid topology
# ══════════════════════════════════════════════════════════════
 
class BusModel(BaseModel):
    """Single node (bus) in the distribution network."""
    bus_id: str
    v_nom_kv: float = 0.4
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None
 
 
class LineModel(BaseModel):
    """Cable connecting two buses."""
    line_id: str
    from_bus: str
    to_bus: str
    length_km: float
    r_ohm_per_km: float
    x_ohm_per_km: float
    max_i_ka: float
 
 
class TransformerModel(BaseModel):
    """MV/LV transformer at the grid head."""
    trafo_id: str
    hv_bus: str
    lv_bus: str
    s_nom_mva: float                    # rated apparent power
    vn_hv_kv: float = 20.0
    vn_lv_kv: float = 0.4
 
 
class GridNetwork(BaseModel):
    """
    Complete low-voltage network topology.
    Produced by grid_model; consumed by rl_engine and dashboard.
    In Phase 1 (manual config): buses and lines are defined by UI.
    In Phase 2 (OSM): produced by data_loader.py from GridCreator.
    """
    network_id: str
    buses: list[BusModel] = Field(default_factory=list)
    lines: list[LineModel] = Field(default_factory=list)
    transformers: list[TransformerModel] = Field(default_factory=list)
    area_name: Optional[str] = None
    household_bus_ids: list[str] = Field(default_factory=list)

    @property
    def n_households(self) -> int:
        return len(self.household_bus_ids)
 
 
# ══════════════════════════════════════════════════════════════
# Household / EV state
# ══════════════════════════════════════════════════════════════
 
class EVState(BaseModel):
    """Live state of a single EV, updated every timestep."""
    agent_id: str
    bus_id: str                          # which bus this EV is connected to
    soc: float = Field(ge=0.0, le=1.0)  # current state of charge
    target_soc: float = EV_TARGET_SOC
    battery_capacity_kwh: float = EV_BATTERY_CAPACITY_KWH
    arrival_step: int  = 0              # timestep index when EV arrives
    departure_step: int = 95            # timestep index when EV must leave
    is_connected: bool = True
 
    @model_validator(mode="after")
    def departure_after_arrival(self) -> "EVState":
        if self.departure_step <= self.arrival_step:
            raise ValueError("departure_step must be > arrival_step")
        return self
 
 

# ══════════════════════════════════════════════════════════════
# RL interface models
# ══════════════════════════════════════════════════════════════
 
class Observation(BaseModel):
    """
    5-dimensional observation vector for one agent.
    Indices match DQNetwork input layer (see constants.OBS_DIM).
    """
    agent_id: str
    soc_progress: float          # current_soc / target_soc  (0–1)
    time_urgency: float          # time_until_departure / episode_length (0–1)
    electricity_price: float     # raw €/kWh
    base_load_kw: float          # inflexible load
    outdoor_temperature_c: float # context feature
 
    def to_array(self) -> list[float]:
        """Return ordered list matching DQNetwork input."""
        return [
            self.soc_progress,
            self.time_urgency,
            self.electricity_price,
            self.base_load_kw,
            self.outdoor_temperature_c,
        ]
 
 
 
class StepResult(BaseModel):
    """Output of GridEnvProtocol.step() for one agent."""
    agent_id: str
    observation: Observation
    reward: float
    done: bool
    truncated: bool = False
    info: dict = Field(default_factory=dict)   # curtailment_kw, etc.
 
 
class PowerFlowResult(BaseModel):
    """
    Outcome of one PyPSA power flow run.
    Produced by grid_model; used to compute rewards and curtailment.
    """
    timestep: int
    transformer_loading_pu: float        # 1.0 = fully loaded
    line_loadings_pu: dict[str, float]   # line_id → loading p.u.
    bus_voltages_pu: dict[str, float]    # bus_id → voltage p.u.
    curtailment_applied: bool
    curtailed_power_kw: dict[str, float] = Field(default_factory=dict)
 
 
# ══════════════════════════════════════════════════════════════
# Episode / simulation result
# ══════════════════════════════════════════════════════════════
 
class EpisodeMetrics(BaseModel):
    """Logged at end of each training episode."""
    episode: int
    mean_episode_reward: float
    soc_satisfaction_rate: float         # fraction of agents hitting target SoC
    curtailment_events: int              # total §14a curtailment events
    transformer_peak_loading_pu: float
    epsilon: float                       # exploration rate at episode end
 
 
class SimResult(BaseModel):
    """
    Full simulation result for one episode.
    Consumed by dashboard for visualisation and export.
    """
    episode: int
    network_id: str
    ev_penetration: float
    timestep_results: list[PowerFlowResult] = Field(default_factory=list)
    final_soc_per_agent: dict[str, float] = Field(default_factory=dict)
    metrics: EpisodeMetrics
 
