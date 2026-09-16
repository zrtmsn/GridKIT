# grid_model/environment.py
# ─────────────────────────────────────────────────────────────
# Multi-device household grid environment.
#
# Each household is equipped with a configurable subset of controllable devices
# (EV / battery / heat pump), each its own RL agent (agent_id = f"{bus}::{device}"),
# plus optional exogenous PV (folded into the meter, surplus exported at the
# feed-in tariff — not an agent). The equipment per home is a HouseholdDevices
# layout (core.models): the map UI edits it; passing only `ev_penetration` builds
# the legacy joint layout (the same homes get the full stack). EVERY home always
# draws base appliance load, whether or not it has controllable devices.
#
# All device-agents at a house SHARE one household-net reward (minimise net
# electricity bill + meet EV SoC + keep heat-pump comfort).
#
# Real exogenous profiles (base load, EV availability, heat-pump demand, PV,
# outdoor temperature) come from grid_model/device_profiles.py (GridCreator /
# pyCity). The day-ahead price stays a synthetic market signal (profiles.py).
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from core.protocols import GridEnvProtocol
from core.models import (
    AGENT_SEP,       # noqa: F401  (re-exported for convenience)
    BatteryAction,
    BatteryState,
    DeviceType,
    EVState,
    GridNetwork,
    HPState,
    HouseholdDevices,
    Observation,
    PowerFlowResult,
    StepResult,
    build_device_layout,
    bus_of,          # noqa: F401  (re-exported)
    device_of,
    make_agent_id,
)
from grid_model.builder import StubNetworkBuilder
from grid_model.surrogate import RadialPowerFlow
from grid_model.profiles import price_profile
from grid_model.device_profiles import DeviceProfileProvider, HouseholdDayProfiles
import core.constants as const


# ══════════════════════════════════════════════════════════════
# Pure device dynamics (no env state — unit-testable)
# ══════════════════════════════════════════════════════════════
def battery_step(soc: float, action_idx: int, *, capacity_kwh: float, max_power_kw: float,
                 efficiency: float, dt: float) -> tuple[float, float]:
    """Apply a battery action.

    Returns (new_soc, bus_power_kw) where bus_power_kw is the power seen at the
    household bus: positive when charging (a load), negative when discharging (a
    source). Charge/discharge are clipped to the usable SoC window and rated power;
    one-way efficiency applies on each leg.
    """
    signed = const.BATTERY_ACTION_TO_KW[action_idx]     # −max / 0 / +max
    if signed > 0:  # CHARGE — energy stored = p·η·dt
        room_kwh = max(0.0, (const.BATTERY_MAX_SOC - soc) * capacity_kwh)
        p = min(signed, max_power_kw, room_kwh / (efficiency * dt) if dt > 0 else 0.0)
        return soc + p * efficiency * dt / capacity_kwh, +p
    if signed < 0:  # DISCHARGE — energy drawn from cell = (p/η)·dt, delivered to bus = p·dt
        avail_kwh = max(0.0, (soc - const.BATTERY_MIN_SOC) * capacity_kwh)
        p = min(max_power_kw, avail_kwh * efficiency / dt if dt > 0 else 0.0)
        return soc - (p / efficiency) * dt / capacity_kwh, -p
    return soc, 0.0


def hp_step(thermal_soc: float, action_idx: int, *, demand_thermal_kw: float,
            cop: float, capacity_kwh: float, dt: float) -> tuple[float, float]:
    """Apply a heat-pump action against the weather-driven heat demand.

    Returns (new_thermal_soc, electric_bus_power_kw). Running adds electric·COP
    thermal energy to the buffer; the demand drains it. The buffer is clipped to
    [0,1] (excess heat when full is wasted — the agent still pays for it).
    """
    electric = const.HP_ACTION_TO_KW[action_idx]         # 0 or rated
    thermal_in = electric * cop
    new_soc = thermal_soc + (thermal_in - demand_thermal_kw) * dt / capacity_kwh
    return float(np.clip(new_soc, 0.0, 1.0)), electric


def curtail_household(controllable_req_kw: float, scale: float) -> float:
    """§14a proportional dimming with the guaranteed floor, per household.

    Returns the delivered aggregate controllable power. Never dims below
    MIN_GUARANTEED_POWER_KW (unless the request itself is smaller).
    """
    if controllable_req_kw <= 0:
        return 0.0
    return min(controllable_req_kw, max(controllable_req_kw * scale, const.MIN_GUARANTEED_POWER_KW))


# ══════════════════════════════════════════════════════════════
# Per-household live state for one episode
# ══════════════════════════════════════════════════════════════
class _Household:
    def __init__(self, bus_id: str, profiles: HouseholdDayProfiles, devices: set[str], has_pv: bool,
                 battery_kwh: float = const.BATTERY_CAPACITY_KWH):
        self.bus_id = bus_id
        self.profiles = profiles
        self.devices = devices        # subset of controllable {ev, battery, hp} this home has
        self.has_pv = has_pv
        # device state objects always exist; only present devices participate/act.
        # EVState.availability comes straight from this episode's sampled profile —
        # real pyCity occupancy data, or a stub-network fallback (see builder).
        self.ev = EVState(
            agent_id=make_agent_id(bus_id, const.DEVICE_EV), bus_id=bus_id,
            soc=0.5, availability=(profiles.ev_available >= 0.5).tolist(),
        )
        self.battery = BatteryState(
            agent_id=make_agent_id(bus_id, const.DEVICE_BATTERY), bus_id=bus_id, capacity_kwh=battery_kwh,
        )
        self.hp = HPState(agent_id=make_agent_id(bus_id, const.DEVICE_HEAT_PUMP), bus_id=bus_id)
        # last available step in the day = the EV's departure deadline for the SoC bonus
        avail = np.flatnonzero(profiles.ev_available >= 0.5)
        self.ev_arrival_step = int(avail[0]) if avail.size else -1
        self.ev_departure_step = int(avail[-1]) if avail.size else -1
        # lagged, locally-observable congestion signals
        self.last_voltage_pu = 1.0
        self.last_net_load_kw = 0.0
        self.last_curtail_ratio = 1.0


class GridEnv(GridEnvProtocol):
    """Multi-device grid environment implementing GridEnvProtocol.

    Agents: for each household, one agent per controllable device it is equipped
    with (ev / battery / hp), per the HouseholdDevices layout. PV is exogenous.
    Call reset() once per episode.
    """

    def __init__(
        self,
        price_scenario: str = "medium",
        ev_penetration: float = 1.0,
        use_surrogate: bool = True,
        builder=None,
        profile_provider: DeviceProfileProvider | None = None,
        device_layout: dict[str, HouseholdDevices] | None = None,
    ):
        self._price_scenario = price_scenario
        self._ev_penetration = ev_penetration
        self._use_surrogate = use_surrogate
        self._provider = profile_provider   # lazily built on first reset if None

        self._network = (builder or StubNetworkBuilder()).build()
        self._surrogate = RadialPowerFlow(self.network)

        # Per-household equipment. Default (no explicit layout): the legacy joint
        # knob — the same round(N·pen) homes, spread evenly, get the full stack.
        all_h = list(self.network.household_bus_ids)
        if device_layout is None:
            device_layout = build_device_layout(
                all_h, ev=ev_penetration, battery=ev_penetration,
                heat_pump=ev_penetration, pv=ev_penetration)
        self._layout: dict[str, HouseholdDevices] = {}
        for bus in all_h:
            self._layout[bus] = device_layout.get(bus) or HouseholdDevices(bus_id=bus)

        # homes with ≥1 controllable device (agents); others draw base load only
        self._active_bus_ids = [b for b in all_h if self._layout[b].controllable]
        # one representative household whose exact per-device state is logged for dashboards
        self._sample_bus_id = self._active_bus_ids[0] if self._active_bus_ids else (all_h[0] if all_h else None)

        self._households: dict[str, _Household] = {}   # keyed by bus_id (ALL homes)
        self._current_step = 0
        self._rng = np.random.default_rng()
        self._price = np.zeros(const.EPISODE_STEPS)

    # ── Metadata ─────────────────────────────────────────────
    @property
    def network(self) -> GridNetwork:
        return self._network

    @property
    def device_layout(self) -> dict[str, HouseholdDevices]:
        return self._layout

    @property
    def agent_ids(self) -> list[str]:
        return [
            make_agent_id(bus, dev)
            for bus, cfg in self._layout.items()
            for dev in cfg.controllable
        ]

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def ev_penetration(self) -> float:
        return self._ev_penetration

    def day_ahead_prices(self) -> list[float]:
        return self._price.tolist()

    # ── Episode exogenous aggregates (for dashboards; valid after reset) ──
    @property
    def episode_base_load_kw(self) -> list[float]:
        """Feeder-mean inflexible base load per step (all homes)."""
        if not self._households:
            return [0.0] * const.EPISODE_STEPS
        return np.mean([hh.profiles.base_load_kw for hh in self._households.values()], axis=0).tolist()

    @property
    def episode_pv_kw(self) -> list[float]:
        """Feeder-total PV generation per step (only PV-equipped homes)."""
        pv_homes = [hh for hh in self._households.values() if hh.has_pv]
        if not pv_homes:
            return [0.0] * const.EPISODE_STEPS
        return np.sum([hh.profiles.pv_gen_kw for hh in pv_homes], axis=0).tolist()

    @property
    def episode_temperature_c(self) -> list[float]:
        """Outdoor temperature per step (shared feeder-wide)."""
        if not self._households:
            return [0.0] * const.EPISODE_STEPS
        return next(iter(self._households.values())).profiles.temperature_c.tolist()

    # ── Episode lifecycle ────────────────────────────────────
    def _ensure_provider(self) -> DeviceProfileProvider:
        if self._provider is None:
            self._provider = DeviceProfileProvider.build()
        return self._provider

    def reset(self, *, seed: int | None = None) -> dict[str, Observation]:
        self._rng = np.random.default_rng(seed)
        self._current_step = 0
        self._price = price_profile(self._rng, self._price_scenario)

        all_h = list(self.network.household_bus_ids)
        provider = self._ensure_provider()
        profiles = provider.sample_episode(len(all_h), self._rng)   # base load for EVERY home
        self._households = {}
        for bus, prof in zip(all_h, profiles):
            cfg = self._layout[bus]
            self._households[bus] = _Household(
                bus, prof, devices=set(cfg.controllable), has_pv=cfg.pv,
                battery_kwh=cfg.battery_kwh or const.BATTERY_CAPACITY_KWH,
            )

        observations: dict[str, Observation] = {}
        for hh in self._households.values():
            for dev in hh.devices:
                observations[make_agent_id(hh.bus_id, dev)] = self._observe(hh, dev)
        return observations

    def step(self, actions: dict[str, int]) -> tuple[dict[str, StepResult], PowerFlowResult]:
        """Advance one 15-minute step. `actions` maps agent_id → device action index."""
        t = self._current_step
        dt = const.TIMESTEP_HOURS

        # 1. Decode actions → per-household requested device powers (kW at the bus).
        #    A home only draws for the devices it actually has; base load is always present.
        req: dict[str, dict] = {}
        for hh in self._households.values():
            p = hh.profiles
            has = hh.devices

            if const.DEVICE_EV in has:
                connected = hh.ev.is_connected_at(t)
                ev_a = int(actions.get(make_agent_id(hh.bus_id, const.DEVICE_EV), 0))
                ev_kw = const.ACTION_TO_KW[ev_a] if (connected and hh.ev.soc < hh.ev.target_soc) else 0.0
            else:
                ev_a, ev_kw = 0, 0.0

            if const.DEVICE_HEAT_PUMP in has:
                hp_a = int(actions.get(make_agent_id(hh.bus_id, const.DEVICE_HEAT_PUMP), 0))
                hp_kw = const.HP_ACTION_TO_KW[hp_a]
            else:
                hp_a, hp_kw = 0, 0.0

            if const.DEVICE_BATTERY in has:
                batt_a = int(actions.get(make_agent_id(hh.bus_id, const.DEVICE_BATTERY), BatteryAction.IDLE))
                _, batt_kw = battery_step(hh.battery.soc, batt_a, capacity_kwh=hh.battery.capacity_kwh,
                                          max_power_kw=hh.battery.max_power_kw,
                                          efficiency=hh.battery.efficiency, dt=dt)
            else:
                batt_a, batt_kw = int(BatteryAction.IDLE), 0.0

            pv = float(p.pv_gen_kw[t]) if hh.has_pv else 0.0
            req[hh.bus_id] = dict(ev_a=ev_a, hp_a=hp_a, batt_a=batt_a,
                                  ev_kw=ev_kw, hp_kw=hp_kw, batt_kw=batt_kw,
                                  base=float(p.base_load_kw[t]), pv=pv)

        # 2. Power flow on the requested household nets — BEFORE curtailment
        max_loading_before, lines_before, volts = self._solve(self._household_nets(req))
        feeder_loadings_before = dict(self._surrogate.last_feeder_loadings)
        hh_feeder = self._surrogate.household_feeder
        hh_path = self._surrogate.household_path_lines
        thresh = const.TRANSFORMER_OVERLOAD_THRESHOLD

        # 3. §14a curtailment — LOCALIZED per feeder: each household is dimmed only by
        #    the overload of ITS OWN feeder (its transformer capacity + its path lines).
        #    Households on healthy feeders keep full power.
        overloaded = (any(v > thresh for v in feeder_loadings_before.values())
                      or any(l > const.LINE_OVERLOAD_THRESHOLD for l in lines_before.values()))
        delivered: dict[str, dict] = {b: dict(r) for b, r in req.items()}
        if overloaded:
            for b, r in req.items():
                local = max(feeder_loadings_before.get(hh_feeder.get(b), 0.0),
                            max((lines_before[l] for l in hh_path.get(b, [])), default=0.0))
                if local <= thresh:
                    continue  # this household's feeder is fine
                creq = r["ev_kw"] + r["hp_kw"] + max(0.0, r["batt_kw"])
                cdel = curtail_household(creq, thresh / local)
                ratio = (cdel / creq) if creq > 1e-9 else 1.0
                delivered[b]["ev_kw"] = r["ev_kw"] * ratio
                delivered[b]["hp_kw"] = r["hp_kw"] * ratio
                delivered[b]["batt_kw"] = (r["batt_kw"] * ratio) if r["batt_kw"] > 0 else r["batt_kw"]
                delivered[b]["ctrl_ratio"] = ratio
            
            # CONSISTENCY FIX: When curtailment was applied, log the BEFORE values
            # (the overload that triggered §14a), not the reduced after values.
            # This ensures: curtailment_applied=True ⇒ overloads are visible in logs!
            max_loading = max_loading_before
            feeder_loadings = feeder_loadings_before
            lines = lines_before
        else:
            # No curtailment → use the original values
            max_loading = max_loading_before
            feeder_loadings = feeder_loadings_before
            lines = lines_before
            
        for b in delivered:
            delivered[b].setdefault("ctrl_ratio", 1.0)

        # 4. Commit device state, compute the shared household reward + observations
        step_results: dict[str, StepResult] = {}
        curtailed_kw: dict[str, float] = {}
        device_power = {const.DEVICE_EV: 0.0, const.DEVICE_BATTERY: 0.0, const.DEVICE_HEAT_PUMP: 0.0, const.DEVICE_PV: 0.0}
        sample_household: dict[str, float] = {}
        for hh in self._households.values():
            d = delivered[hh.bus_id]
            has = hh.devices
            net_kw = d["base"] + d["ev_kw"] + d["hp_kw"] + d["batt_kw"] - d["pv"]
            device_power[const.DEVICE_EV] += d["ev_kw"]
            device_power[const.DEVICE_HEAT_PUMP] += d["hp_kw"]
            device_power[const.DEVICE_BATTERY] += d["batt_kw"]   # signed
            device_power[const.DEVICE_PV] -= d["pv"]             # generation (≤0)

            # EV SoC
            if const.DEVICE_EV in has and d["ev_kw"] > 0:
                hh.ev.soc = min(1.0, hh.ev.soc + d["ev_kw"] * dt / hh.ev.battery_capacity_kwh)
            # Battery SoC (recompute with the *delivered* charge power; discharge unchanged)
            if const.DEVICE_BATTERY in has:
                if d["batt_kw"] > 0 and d["batt_kw"] < req[hh.bus_id]["batt_kw"] - 1e-9:
                    # charge was curtailed → apply the delivered charge directly
                    hh.battery.soc = min(
                        const.BATTERY_MAX_SOC,
                        hh.battery.soc + d["batt_kw"] * hh.battery.efficiency * dt / hh.battery.capacity_kwh,
                    )
                else:
                    hh.battery.soc, _ = battery_step(
                        hh.battery.soc, d["batt_a"], capacity_kwh=hh.battery.capacity_kwh,
                        max_power_kw=hh.battery.max_power_kw, efficiency=hh.battery.efficiency, dt=dt)
            # Heat-pump thermal buffer
            if const.DEVICE_HEAT_PUMP in has:
                demand_thermal = float(hh.profiles.hp_load_kw[t]) * const.HP_COP
                hh.hp.thermal_soc, _ = hp_step(hh.hp.thermal_soc, d["hp_a"], demand_thermal_kw=demand_thermal,
                                               cop=const.HP_COP, capacity_kwh=hh.hp.thermal_capacity_kwh, dt=dt)

            if hh.bus_id == self._sample_bus_id:
                sample_household = {
                    "ev_kw": d["ev_kw"], "battery_kw": d["batt_kw"], "hp_kw": d["hp_kw"], "pv_kw": d["pv"],
                    "ev_available": 1.0 if (const.DEVICE_EV in has and hh.profiles.ev_available[t] >= 0.5) else 0.0,
                    "ev_soc": hh.ev.soc, "battery_soc": hh.battery.soc, "hp_soc": hh.hp.thermal_soc,
                }

            reward, info = self._reward(hh, net_kw, t, d)
            # stash lagged congestion signals for the next observation
            hh.last_voltage_pu = float(volts.get(hh.bus_id, 1.0))
            hh.last_net_load_kw = net_kw
            hh.last_curtail_ratio = d["ctrl_ratio"]

            ctrl_req = req[hh.bus_id]["ev_kw"] + req[hh.bus_id]["hp_kw"] + max(0.0, req[hh.bus_id]["batt_kw"])
            ctrl_del = d["ev_kw"] + d["hp_kw"] + max(0.0, d["batt_kw"])
            if ctrl_req - ctrl_del > 1e-9:
                curtailed_kw[hh.bus_id] = ctrl_req - ctrl_del

            for dev in has:
                aid = make_agent_id(hh.bus_id, dev)
                step_results[aid] = StepResult(
                    agent_id=aid, observation=self._observe(hh, dev), reward=reward,
                    done=t >= const.EPISODE_STEPS - 1, truncated=False, info=info,
                )

        pf = PowerFlowResult(
            timestep=t, transformer_loading_pu=max_loading, transformer_loadings_pu=feeder_loadings,
            line_loadings_pu=lines, bus_voltages_pu=volts, curtailment_applied=overloaded,
            curtailed_power_kw=curtailed_kw, device_power_kw=device_power, sample_household=sample_household,
        )
        self._current_step += 1
        return step_results, pf

    # ── Physics ──────────────────────────────────────────────
    def _household_nets(self, table: dict[str, dict]) -> dict[str, float]:
        """bus_id → net active power (MW). Every home is in `table` (base load always)."""
        nets_mw: dict[str, float] = {}
        for bus_id in self.network.household_bus_ids:
            r = table.get(bus_id)
            net_kw = (r["base"] + r["ev_kw"] + r["hp_kw"] + r["batt_kw"] - r["pv"]) if r else 0.0
            nets_mw[bus_id] = net_kw / 1000.0
        return nets_mw

    def _solve(self, bus_load_mw: dict[str, float]):
        return self._surrogate.solve(bus_load_mw)

    # ── Observation / reward ─────────────────────────────────
    def _device_soc(self, hh: _Household, device: str) -> tuple[float, float]:
        """(soc_progress, time_urgency) for a device — its OWN state."""
        if device == const.DEVICE_EV:
            urg = max(0.0, (hh.ev_departure_step - self._current_step) / const.EPISODE_STEPS) \
                if hh.ev_departure_step >= 0 else 0.0
            return hh.ev.soc / hh.ev.target_soc, urg
        if device == const.DEVICE_BATTERY:
            return hh.battery.soc, 0.0
        return hh.hp.thermal_soc, 0.0   # heat pump

    def _observe(self, hh: _Household, device: str) -> Observation:
        t = min(self._current_step, const.EPISODE_STEPS - 1)
        soc_progress, urgency = self._device_soc(hh, device)
        return Observation(
            agent_id=make_agent_id(hh.bus_id, device),
            device_type=DeviceType(device),
            soc_progress=soc_progress,
            time_urgency=urgency,
            electricity_price=float(self._price[t]),
            base_load_kw=float(hh.profiles.base_load_kw[t]),
            outdoor_temperature_c=float(hh.profiles.temperature_c[t]),
            local_voltage_pu=hh.last_voltage_pu,
            recent_curtailment_ratio=hh.last_curtail_ratio,
            feed_in_price=const.FEED_IN_TARIFF_EUR_KWH,
            pv_generation_kw=float(hh.profiles.pv_gen_kw[t]) if hh.has_pv else 0.0,
            net_household_load_kw=hh.last_net_load_kw,
            time_of_day=self._current_step / const.EPISODE_STEPS,
        )

    def _reward(self, hh: _Household, net_kw: float, t: int, d: dict) -> tuple[float, dict]:
        """Household-net reward shared by all device-agents at this house."""
        dt = const.TIMESTEP_HOURS
        # net electricity bill: import at retail, export (surplus) at feed-in
        price = float(self._price[min(t, const.EPISODE_STEPS - 1)])
        if net_kw >= 0:
            bill = price * net_kw * dt
        else:
            bill = -const.FEED_IN_TARIFF_EUR_KWH * (-net_kw) * dt
        reward = -bill * const.REWARD_ELECTRICITY_COST_WEIGHT

        info: dict = {"net_kw": net_kw, "bill_eur": bill}

        # heat-pump comfort: per-step penalty proportional to how far below the floor (HP homes only)
        if const.DEVICE_HEAT_PUMP in hh.devices:
            deficit = max(0.0, hh.hp.comfort_min_soc - hh.hp.thermal_soc)
            if deficit > 0:
                reward += const.REWARD_HP_COMFORT_MISS_PENALTY * deficit
            info["hp_thermal_soc"] = hh.hp.thermal_soc

        # EV SoC terminal penalty at the departure deadline, QUADRATIC in the shortfall
        # (EV homes only). Not a cliff — arriving 1% short is a minor annoyance and must
        # score like one, or the agent pays any price to cross an arbitrary line and price
        # signals stop mattering. Not linear either — a linear penalty is exploitable, and
        # measurably was: the agent traded 13.9% of the charge for a smaller bill. Squaring
        # keeps small deviations cheap and makes deep undercharging prohibitive.
        if const.DEVICE_EV in hh.devices and hh.ev_departure_step >= 0 and t == hh.ev_departure_step:
            target = max(hh.ev.target_soc, 1e-9)
            shortfall = min(1.0, max(0.0, (target - hh.ev.soc) / target))
            satisfied = hh.ev.soc >= hh.ev.target_soc
            reward += const.REWARD_SOC_MISS_PENALTY * shortfall ** 2
            if satisfied:
                reward += const.REWARD_SOC_COMPLETION_BONUS
            info.update(ev_terminal=True, ev_final_soc=hh.ev.soc, ev_satisfied=satisfied,
                        ev_shortfall=shortfall)
        return reward, info
