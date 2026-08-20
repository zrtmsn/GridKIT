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
    EPISODE_STEPS,
    EV_BATTERY_CAPACITY_KWH,
    EV_TARGET_SOC,
    BATTERY_CAPACITY_KWH,
    BATTERY_MAX_POWER_KW,
    BATTERY_EFFICIENCY,
    BATTERY_INITIAL_SOC,
    HP_THERMAL_CAPACITY_KWH,
    HP_INITIAL_THERMAL_SOC,
    HP_COMFORT_MIN_SOC,
    DEVICE_EV,
    DEVICE_BATTERY,
    DEVICE_HEAT_PUMP,
    CONTROLLABLE_DEVICE_TYPES,
)


# ══════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════

class ChargingAction(IntEnum):
    OFF  = 0   # 0.0 kW
    HALF = 1   # 3.7 kW
    FULL = 2   # 7.4 kW


# One controllable device = one RL agent. All agents of the same DeviceType
# share a single policy (constants.DEVICE_*). Action enums below are the
# per-device discrete action spaces (values are RLlib Discrete indices).
class DeviceType(StrEnum):
    EV        = "ev"
    BATTERY   = "battery"
    HEAT_PUMP = "hp"
    PV        = "pv"


class BatteryAction(IntEnum):
    DISCHARGE = 0   # supply the house from the battery (−BATTERY_MAX_POWER_KW)
    IDLE      = 1   # hold
    CHARGE    = 2   # store into the battery (+BATTERY_MAX_POWER_KW)


class HPAction(IntEnum):
    OFF  = 0   # compressor off this step
    HEAT = 1   # run (draw HP_RATED_ELECTRIC_KW, deliver ×COP thermal)


# PV is exogenous generation (folded into the battery's store-vs-export decision),
# not a controllable agent — so there is no PVAction. See constants.DEVICE_PV.


class PriceScenario(StrEnum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ── Agent-id encoding (shared contract: one agent per controllable device) ──
# agent_id = f"{bus_id}{AGENT_SEP}{device}", e.g. "household_3::battery". The
# device suffix selects the shared policy. Lives in core so every module
# (grid_model, rl_engine, scenarios) uses the same convention without importing
# each other.
AGENT_SEP: str = "::"


def make_agent_id(bus_id: str, device: str) -> str:
    return f"{bus_id}{AGENT_SEP}{device}"


def device_of(agent_id: str) -> str:
    return agent_id.split(AGENT_SEP, 1)[1]


def bus_of(agent_id: str) -> str:
    return agent_id.split(AGENT_SEP, 1)[0]


# ══════════════════════════════════════════════════════════════
# Grid topology
# ══════════════════════════════════════════════════════════════

class BusModel(BaseModel):
    """Single node (bus) in the distribution network."""
    bus_id: str
    v_nom_kv: float = 0.4       # nominal voltage in kV (LV grid = 0.4 kV)
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None


class LineModel(BaseModel):
    """Cable connecting two buses."""
    line_id: str
    from_bus: str
    to_bus: str
    length_km: float
    r_ohm_per_km: float         # resistance — causes active power loss
    x_ohm_per_km: float         # reactance — causes reactive power loss
    max_i_ka: float             # maximum current in kilo-amperes before overload


class TransformerModel(BaseModel):
    """MV/LV transformer at the grid head."""
    trafo_id: str
    hv_bus: str                 # high-voltage side bus (medium voltage, 20 kV)
    lv_bus: str                 # low-voltage side bus (0.4 kV, feeds households)
    s_nom_mva: float            # rated apparent power in MVA (e.g. 0.16 = 160 kVA)
    vn_hv_kv: float = 20.0     # nominal voltage high-voltage side
    vn_lv_kv: float = 0.4      # nominal voltage low-voltage side


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
    household_bus_ids: list[str] = Field(default_factory=list)  # bus IDs that are residential connection points (from GridCreator buses_df)
    # Optional real data from GridCreator, one representative day at episode
    # resolution (length == EPISODE_STEPS). Absent/empty for stub networks —
    # grid_model falls back to synthetic profiles and always-connected EVs.
    household_load_profile_kw: dict[str, list[float]] = Field(default_factory=dict)
    ev_availability: dict[str, list[bool]] = Field(default_factory=dict)  # household_bus_id -> plugged-in per step

    @property
    def n_households(self) -> int:
        return len(self.household_bus_ids)


class FeederSummary(BaseModel):
    """
    One transformer's radial feeder within a (possibly multi-transformer)
    GridNetwork — enough for a UI to list/highlight feeder options on a map
    without re-deriving the bus membership itself.
    Produced by grid_model.builder.OSMNetworkBuilder.list_feeders().
    """
    trafo_id: str
    household_count: int
    bus_ids: list[str]  # every bus (household or not) reachable from this transformer


# ══════════════════════════════════════════════════════════════
# Household / EV state
# ══════════════════════════════════════════════════════════════

class EVState(BaseModel):
    """Live state of a single EV, updated every timestep."""
    agent_id: str
    bus_id: str                          # which bus this EV is connected to
    soc: float = Field(ge=0.0, le=1.0)  # state of charge: 0.0 = empty, 1.0 = full
    target_soc: float = EV_TARGET_SOC   # SoC the agent must reach by the end of the episode
    battery_capacity_kwh: float = EV_BATTERY_CAPACITY_KWH
    # Plugged-in/away per episode step — same structure as GridCreator's own
    # Link.p_max_pu availability series (core/protocols.py / grid_model.builder).
    # Defaults to "always connected" for networks without real availability data.
    # Heterogeneous synthetic connection windows (grid_model.device_profiles)
    # populate this from each household's sampled arrival/departure just like
    # real GridCreator data would — one representation for both sources.
    availability: list[bool] = Field(default_factory=lambda: [True] * EPISODE_STEPS)

    @model_validator(mode="after")
    def availability_matches_episode_length(self) -> "EVState":
        if len(self.availability) != EPISODE_STEPS:
            raise ValueError(f"availability must have exactly {EPISODE_STEPS} entries, got {len(self.availability)}")
        return self

    def is_connected_at(self, step: int) -> bool:
        return self.availability[step]


class BatteryState(BaseModel):
    """Live state of a home battery, updated every timestep.

    Sign convention (see constants.BATTERY_ACTION_TO_KW): positive power charges
    the battery (drawn from PV/grid), negative discharges it (supplies the house).
    Round-trip losses apply via BATTERY_EFFICIENCY on both legs.
    """
    agent_id: str
    bus_id: str
    soc: float = Field(BATTERY_INITIAL_SOC, ge=0.0, le=1.0)
    capacity_kwh: float = BATTERY_CAPACITY_KWH
    max_power_kw: float = BATTERY_MAX_POWER_KW
    efficiency: float = BATTERY_EFFICIENCY


class HPState(BaseModel):
    """Live state of a heat pump, modelled as a thermal buffer (house inertia).

    ``thermal_soc`` is the normalized buffer level [0,1]. Running the HP charges
    it (electric × COP), the weather-driven heat demand drains it. Comfort is met
    while ``thermal_soc`` stays ≥ ``comfort_min_soc``.
    """
    agent_id: str
    bus_id: str
    thermal_soc: float = Field(HP_INITIAL_THERMAL_SOC, ge=0.0, le=1.0)
    thermal_capacity_kwh: float = HP_THERMAL_CAPACITY_KWH
    comfort_min_soc: float = HP_COMFORT_MIN_SOC


class HouseholdDevices(BaseModel):
    """Which controllable devices a household is equipped with (+ exogenous PV).

    This is the per-household layout the map UI edits and the environment consumes:
    a home only spawns agents for the devices it actually has. Every home always
    draws base appliance load regardless.
    """
    bus_id: str
    ev: bool = False
    battery: bool = False
    heat_pump: bool = False
    pv: bool = False
    pv_kwp: Optional[float] = None   # optional PV size override (else provider-sampled)

    @property
    def controllable(self) -> list[str]:
        """Device-type ids present, ordered as CONTROLLABLE_DEVICE_TYPES (→ RL agents)."""
        present = {DEVICE_EV: self.ev, DEVICE_BATTERY: self.battery, DEVICE_HEAT_PUMP: self.heat_pump}
        return [d for d in CONTROLLABLE_DEVICE_TYPES if present.get(d)]


def _spread(n_total: int, frac: float) -> set[int]:
    """Indices of round(frac·n) homes, spread evenly across [0, n) (deterministic)."""
    k = max(0, min(n_total, round(n_total * frac)))
    if k <= 0:
        return set()
    if k >= n_total:
        return set(range(n_total))
    if k == 1:
        return {n_total // 2}
    return {round(i * (n_total - 1) / (k - 1)) for i in range(k)}


def build_device_layout(
    household_bus_ids: list[str],
    *,
    ev: float = 1.0,
    battery: float = 1.0,
    heat_pump: float = 1.0,
    pv: float = 1.0,
) -> dict[str, "HouseholdDevices"]:
    """Assign each device type to a fraction of households, spread evenly.

    Per-device penetration knobs (the map-UI sliders). Passing equal fractions
    reproduces the legacy joint behaviour (the same homes get the full stack).
    """
    n = len(household_bus_ids)
    sel = {
        DEVICE_EV: _spread(n, ev),
        DEVICE_BATTERY: _spread(n, battery),
        DEVICE_HEAT_PUMP: _spread(n, heat_pump),
        "pv": _spread(n, pv),
    }
    return {
        bus: HouseholdDevices(
            bus_id=bus,
            ev=i in sel[DEVICE_EV],
            battery=i in sel[DEVICE_BATTERY],
            heat_pump=i in sel[DEVICE_HEAT_PUMP],
            pv=i in sel["pv"],
        )
        for i, bus in enumerate(household_bus_ids)
    }


# ══════════════════════════════════════════════════════════════
# RL interface models
# ══════════════════════════════════════════════════════════════

class Observation(BaseModel):
    """
    7-dimensional observation vector for one agent (see constants.OBS_DIM).
    All fields are raw/semantic; the RLlib wrapper normalizes them to [0,1].

    The last two are *lagged* congestion signals a smart meter can plausibly observe
    (local voltage drop, and whether the device was dimmed). They give a purely
    self-interested agent a channel to perceive — and thus avoid — grid stress,
    without any cooperative/grid-protective term being hand-coded into the reward.
    """
    agent_id: str
    soc_progress: float          # own device progress-to-goal (EV soc/target, battery soc, HP thermal soc; PV≈1)
    time_urgency: float          # time_until_deadline / episode_length (EV/HP; 0 for battery/PV) (0–1)
    electricity_price: float     # current retail tariff in €/kWh
    base_load_kw: float          # non-controllable household consumption (BDEW H0 profile)
    outdoor_temperature_c: float # ambient temperature (real TRY series in the multi-device env)
    local_voltage_pu: float = 1.0          # previous-step voltage at this household's bus (p.u.)
    recent_curtailment_ratio: float = 1.0  # previous-step delivered/requested power (1=unaffected, 0=fully cut)

    # ── Multi-device fields (populated by the multi-device env; default-safe
    #    so the legacy EV-only path and its 7-dim to_array() are unaffected) ──
    device_type: DeviceType = DeviceType.EV
    feed_in_price: float = 0.0             # PV feed-in tariff (€/kWh) this episode
    pv_generation_kw: float = 0.0          # household PV generation this step
    net_household_load_kw: float = 0.0     # lagged household net = consumption − PV (kW)
    time_of_day: float = 0.0               # step / EPISODE_STEPS (0–1); timing signal for battery/PV

    def to_array(self) -> list[float]:
        """Legacy 7-dim EV-only vector (matches OBS_DIM). Kept for the EV path."""
        return [
            self.soc_progress,
            self.time_urgency,
            self.electricity_price,
            self.base_load_kw,
            self.outdoor_temperature_c,
            self.local_voltage_pu,
            self.recent_curtailment_ratio,
        ]

    def to_array_multidevice(self) -> list[float]:
        """Unified 10-dim vector (matches OBS_DIM_MULTIDEVICE) used by every
        device-type agent. Layout is uniform; soc_progress/time_urgency carry the
        agent's OWN device state. The multi-device env switches to this."""
        return [
            self.soc_progress,
            self.time_urgency,
            self.electricity_price,
            self.feed_in_price,
            self.net_household_load_kw,
            self.pv_generation_kw,
            self.outdoor_temperature_c,
            self.local_voltage_pu,
            self.recent_curtailment_ratio,
            self.time_of_day,
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
    Outcome of one power flow run (surrogate or PyPSA).
    Produced by grid_model; used to compute rewards and curtailment.
    """
    timestep: int
    transformer_loading_pu: float        # MAX feeder loading (fraction of rated) — 1.0 = fully loaded, >1.0 = overload
    transformer_loadings_pu: dict[str, float] = Field(default_factory=dict)  # feeder_id → loading (multi-feeder grids)
    line_loadings_pu: dict[str, float]   # line_id → load fraction of rated capacity (p.u. = per unit)
    bus_voltages_pu: dict[str, float]    # bus_id → voltage as fraction of nominal (1.0 = 0.4 kV)
    bus_load_mw: dict[str, float] = Field(default_factory=dict)  # household bus_id → actual (post-curtailment) load in MW
    curtailment_applied: bool            # True if §14a power reduction was triggered this timestep
    curtailed_power_kw: dict[str, float] = Field(default_factory=dict)  # agent_id → kW that was cut
    device_power_kw: dict[str, float] = Field(default_factory=dict)     # device type → feeder-aggregate delivered kW
                                                                        # (ev/hp ≥0 load; battery signed: + charge / − discharge; pv ≤0 generation)
    sample_household: dict[str, float] = Field(default_factory=dict)    # ONE representative home's exact state this step:
                                                                        # ev_kw, battery_kw, hp_kw, pv_kw, ev_available, ev_soc, battery_soc, hp_soc


# ══════════════════════════════════════════════════════════════
# Episode / simulation result
# ══════════════════════════════════════════════════════════════

class EpisodeMetrics(BaseModel):
    """Logged at end of each evaluation episode. Not populated during training."""
    episode: int
    mean_episode_reward: float
    soc_satisfaction_rate: float         # fraction of agents that reached target SoC by departure (0–1)
    curtailment_events: int              # number of timesteps where §14a curtailment was triggered
    transformer_peak_loading_pu: float   # highest transformer load seen during the episode (p.u.)
    # What the household actually PAYS over the episode (€, import at retail minus export
    # at feed-in). A grid-friendly policy nobody would install is worthless, so the
    # adoption case needs the bill as a first-class outcome, not just as a reward term.
    mean_household_bill_eur: float = 0.0

    # What the OTHER two controllable devices were actually doing — without these the
    # scenario comparison is silently EV-only even though battery/HP run every episode.
    # Fraction of (HP-equipped-household, step) pairs where the thermal buffer stayed at
    # or above its comfort floor (mirrors soc_satisfaction_rate, but per-step since HP
    # has no single terminal deadline the way an EV departure does).
    hp_comfort_satisfaction_rate: float = 0.0
    # Feeder-aggregate battery energy cycled over the episode (kWh) — charge and discharge
    # kept separate rather than netted, since a battery that never moves and one that
    # charges 5 kWh then discharges 5 kWh both net to zero but are very different behaviors.
    battery_charge_kwh: float = 0.0
    battery_discharge_kwh: float = 0.0

    # WHERE the stress was, not just how often. §14a dimming is applied per feeder, so a
    # bare event count hides whether one weak feeder caused everything or the whole grid
    # is marginal. Feeders are listed in full (there are few, and a feeder sitting at
    # 0.95 pu is worth seeing); lines only when they actually tripped, since a village
    # has thousands and listing them all would swamp the record.
    feeder_overload_steps: dict[str, int] = Field(default_factory=dict)      # feeder_id → steps over threshold
    feeder_peak_loading_pu: dict[str, float] = Field(default_factory=dict)   # feeder_id → worst loading seen
    line_overload_steps: dict[str, int] = Field(default_factory=dict)        # line_id → steps over threshold
    line_peak_loading_pu: dict[str, float] = Field(default_factory=dict)     # line_id → worst loading seen
    # DQN only: exploration rate at episode end (1.0 = fully random, 0.0 = greedy); None for IPPO
    epsilon: Optional[float] = None
    # IPPO only: mean policy entropy across all agents at episode end; None for DQN
    entropy: Optional[float] = None


class SimResult(BaseModel):
    """
    Full simulation result for one episode.
    Consumed by dashboard for visualisation and export.
    """
    episode: int
    network_id: str
    ev_penetration: float                # fraction of households with an EV (e.g. 0.20 = 20%)
    timestep_results: list[PowerFlowResult] = Field(default_factory=list)  # one entry per timestep (EPISODE_STEPS total)
    final_soc_per_agent: dict[str, float] = Field(default_factory=dict)    # agent_id → SoC at departure
    metrics: EpisodeMetrics

