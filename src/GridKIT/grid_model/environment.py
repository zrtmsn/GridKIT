# grid_model/environment.py
import random as _random
import zlib

from core.protocols import GridEnvProtocol
from core.models import ChargingAction, EVState, GridNetwork, Observation, PowerFlowResult, StepResult
from grid_model.builder import StubNetworkBuilder
from grid_model.network import build_pypsa_network
from grid_model.surrogate import RadialPowerFlow
from grid_model.bdew_h0 import build_h0_profile
from grid_model.ev_availability import synthetic_ev_availability
import core.constants as const


class GridEnv(GridEnvProtocol):
    """
    Power grid simulation environment implementing GridEnvProtocol from core
    call reset() once after initiating
    implementation details:
        EV availability is a per-step plugged-in/away array — the same
        structure GridCreator itself uses (Link.p_max_pu). Networks without
        real availability data (including the stub network) get a synthetic
        occupancy-based profile instead of a flat always-connected default —
        deterministic per bus (seeded from the bus id), but not uniform.
        There is no single arrival/departure event any more: charging
        requests are gated every step by availability, and the SoC target
        is evaluated once, at the last step of the episode.
        every household has an ev and a base load
        stub network defined in data/stub_network.json
    """

    def __init__(self, use_surrogate: bool = True, network: GridNetwork | None = None):
        self._use_surrogate = use_surrogate
        self._network = network or StubNetworkBuilder().build()
        self._surrogate = RadialPowerFlow(self._network)
        self._pypsa_network_template = build_pypsa_network(self.network)
        self._evs: dict[str, EVState] = {} #key = agent_id
        self._current_step = 0
        self._load_multiplier: float = 1.0  # set per episode in reset()
        # Build H0 profile once — fallback base load for buses without a real profile
        self._h0_profile_kw = build_h0_profile(self._pypsa_network_template.snapshots)

        for x in self.network.household_bus_ids:
            availability = self.network.ev_availability.get(x)
            if not availability:
                # Stable per-bus seed (not Python's hash(), which is
                # randomized per-process) so the same network always
                # produces the same synthetic availability.
                seed = zlib.crc32(x.encode())
                availability = synthetic_ev_availability(const.SYNTHETIC_EV_HOUSEHOLD_SIZE, seed=seed)
            current_ev = EVState(
            agent_id=x,
            bus_id=x,
            soc=const.EV_INITIAL_SOC_MEAN,
            availability=availability,
            )
            self._evs[x] = current_ev
            self._pypsa_network_template.add("Load", f"ev_{x}", bus=x, p_set=0)
            self._pypsa_network_template.add("Load", f"base_{x}", bus=x, p_set=0)

    def _base_load_kw(self, bus_id: str, step: int) -> float:
        profile = self.network.household_load_profile_kw.get(bus_id)
        if profile is not None:
            return profile[step]
        return self._h0_profile_kw.iloc[step] * self._load_multiplier

    @property
    def network(self) -> GridNetwork:
        return self._network

    @property
    def agent_ids(self) -> list[str]:
        return list(self.network.household_bus_ids)

    @property
    def current_step(self) -> int:
        return self._current_step
    
    def reset(self, *, seed: int | None = None):
        """
        resets the network to step 0
        implementation details:
            load multiplier sampled per episode from [LOAD_MULTIPLIER_MIN, LOAD_MULTIPLIER_MAX]
            (only affects buses without a real GridCreator profile)
            EV SoC reset to mean; availability fixed for the episode (see __init__)
        """
        rng = _random.Random(seed)
        self._load_multiplier = rng.uniform(const.LOAD_MULTIPLIER_MIN, const.LOAD_MULTIPLIER_MAX)

        self._pypsa_network = self._pypsa_network_template.copy()
        self._current_step = 0
        for x in self._evs.values():
            x.soc = const.EV_INITIAL_SOC_MEAN
        observations: dict[str, Observation] = {}
        for x in self.network.household_bus_ids:
            ev = self._evs[x]
            current = Observation(
            agent_id = x,
            soc_progress = ev.soc,
            time_urgency = (const.EPISODE_STEPS - 1 - self._current_step) / const.EPISODE_STEPS,
            electricity_price = 1.0,
            base_load_kw = self._base_load_kw(x, self._current_step),
            outdoor_temperature_c = 20.0
            )
            observations[x] = current

        return observations


    def step(self, actions: dict[str, ChargingAction]) -> tuple[dict[str, StepResult], PowerFlowResult]:
        """
        advances the network by one step, applying the given strategy
        implementation details:
            base loads from a real GridCreator profile where the network provides
            one, else the BDEW H0 profile scaled by per-episode load multiplier
            EV charging requests are forced to 0 kW while the EV isn't connected
            (real GridCreator availability where present, else synthetic per __init__)
            price set to constant 1.0 for now
            curtailment based on max load of lines and trafo, with multiplier defined in const
            reward calculation:
                reward = -electricity_cost * REWARD_ELECTRICITY_COST_WEIGHT
                reward += REWARD_SOC_COMPLETION_BONUS if soc >= target at the end of the episode
                reward += REWARD_SOC_MISS_PENALTY if end of episode and soc < target

        """
        # 1. Collect requested EV power and base load
        base_load_kw = {
            bus_id: self._base_load_kw(bus_id, self._current_step)
            for bus_id in self.network.household_bus_ids
        }
        ev_kw: dict[str, float] = {
            agent_id: const.ACTION_TO_KW[actions.get(agent_id, ChargingAction.OFF)]
                if ev.is_connected_at(self._current_step) else 0.0
            for agent_id, ev in self._evs.items()
        }

        # 2. Power flow with requested loads
        bus_load_mw = {
            bus_id: (base_load_kw[bus_id] + ev_kw.get(bus_id, 0.0)) / 1000.0
            for bus_id in self.network.household_bus_ids
        }
        trafo_loading, line_loadings, bus_voltages = self._solve(bus_load_mw)

        # 3. §14a curtailment
        curtailment_applied = False
        curtailed_power_kw: dict[str, float] = {}
        delivered_kw = dict(ev_kw)

        overloaded = (
            trafo_loading > const.TRANSFORMER_OVERLOAD_THRESHOLD or
            any(l > const.LINE_OVERLOAD_THRESHOLD for l in line_loadings.values())
        )

        if overloaded:
            curtailment_applied = True
            for agent_id in self._evs:
                original_kw = ev_kw[agent_id]
                capped_kw = min(original_kw, const.MAX_CONTROLLED_POWER_KW)
                curtailed_kw = original_kw - capped_kw
                if curtailed_kw > 0:
                    curtailed_power_kw[agent_id] = curtailed_kw
                delivered_kw[agent_id] = capped_kw
            bus_load_mw = {
                bus_id: (base_load_kw[bus_id] + delivered_kw.get(bus_id, 0.0)) / 1000.0
                for bus_id in self.network.household_bus_ids
            }
            trafo_loading, line_loadings, bus_voltages = self._solve(bus_load_mw)

        for agent_id in self._evs:
            energy_kwh = delivered_kw[agent_id] * const.TIMESTEP_HOURS
            self._evs[agent_id].soc = min(
                1.0,
                self._evs[agent_id].soc + energy_kwh / const.EV_BATTERY_CAPACITY_KWH
            )

        # 5. Compute rewards and build results
        is_last_step = self._current_step >= const.EPISODE_STEPS - 1
        step_results: dict[str, StepResult] = {}
        for agent_id, ev in self._evs.items():
            actual_kw = delivered_kw[agent_id]
            electricity_cost = actual_kw * const.TIMESTEP_HOURS * 1.0  # price placeholder
            reward = -electricity_cost * const.REWARD_ELECTRICITY_COST_WEIGHT

            if is_last_step:
                if ev.soc >= ev.target_soc:
                    reward += const.REWARD_SOC_COMPLETION_BONUS
                else:
                    reward += const.REWARD_SOC_MISS_PENALTY

            obs = Observation(
                agent_id=agent_id,
                soc_progress=ev.soc / ev.target_soc,
                time_urgency=max(0, (const.EPISODE_STEPS - 1 - self._current_step) / const.EPISODE_STEPS),
                electricity_price=1.0,  # TODO: price scenario
                base_load_kw=base_load_kw[agent_id],
                outdoor_temperature_c=20.0,
            )

            step_results[agent_id] = StepResult(
                agent_id=agent_id,
                observation=obs,
                reward=reward,
                done=is_last_step,
                truncated=False,
                info={"curtailed_kw": curtailed_power_kw.get(agent_id, 0.0)}
            )

        power_flow_result = PowerFlowResult(
            timestep=self._current_step,
            transformer_loading_pu=trafo_loading,
            line_loadings_pu=line_loadings,
            bus_voltages_pu=bus_voltages,
            bus_load_mw=bus_load_mw,
            curtailment_applied=curtailment_applied,
            curtailed_power_kw=curtailed_power_kw
        )

        self._current_step += 1
        return step_results, power_flow_result

    # ── Physics backend ──────────────────────────────────────
    def _solve(self, bus_load_mw: dict[str, float]):
        """Return (trafo_loading_pu, {line: loading_pu}, {bus: voltage_pu})."""
        if self._use_surrogate:
            return self._surrogate.solve(bus_load_mw)
        # Slow PyPSA LPF path — for validation only
        snapshot = self._pypsa_network.snapshots[self._current_step]
        for bus_id, mw in bus_load_mw.items():
            self._pypsa_network.loads.at[f"base_{bus_id}", "p_set"] = mw
        self._pypsa_network.lpf(snapshots=snapshot)
        trafo = self._pypsa_network.transformers.index[0]
        trafo_loading = abs(self._pypsa_network.transformers_t.p0.at[snapshot, trafo]) / \
            self._pypsa_network.transformers.at[trafo, "s_nom"]
        line_loadings = {
            line_id: abs(self._pypsa_network.lines_t.p0.at[snapshot, line_id]) /
            self._pypsa_network.lines.at[line_id, "s_nom"]
            for line_id in self._pypsa_network.lines.index
        }
        voltages = {
            b: float(self._pypsa_network.buses_t.v_mag_pu.at[snapshot, b])
            for b in self._pypsa_network.buses.index
        }
        return float(trafo_loading), {k: float(v) for k, v in line_loadings.items()}, voltages
