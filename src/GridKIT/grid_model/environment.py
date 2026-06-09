# grid_model/environment.py
import random as _random

from core.protocols import GridEnvProtocol
from core.models import ChargingAction, EVState, GridNetwork, Observation, PowerFlowResult, StepResult
from grid_model.builder import StubNetworkBuilder
from grid_model.network import build_pypsa_network
from grid_model.bdew_h0 import get_base_load_kw
import core.constants as const


class GridEnv(GridEnvProtocol):
    """
    Power grid simulation environment implementing GridEnvProtocol from core
    call reset() once after initiating
    implementation details:
        ev arrival and departure set to 0 and 95 for now due to model validator failing
        every household has an ev and a base load
        stub network defined in data/stub_network.json
    """

    def __init__(self):
        self._network = StubNetworkBuilder().build()
        self._pypsa_network_template = build_pypsa_network(self.network)
        self._evs: dict[str, EVState] = {} #key = agent_id
        self._current_step = 0
        self._load_multiplier: float = 1.0  # set per episode in reset()

        for x in self.network.household_bus_ids:
            current_ev = EVState(
            agent_id=x,
            bus_id=x,
            soc=const.EV_INITIAL_SOC_MEAN,
            # TODO: find solution for correct arrival and departure steps
            arrival_step=0,
            departure_step=const.EPISODE_STEPS - 1
            )
            self._evs[x] = current_ev
            self._pypsa_network_template.add("Load", f"ev_{x}", bus=x, p_set=0)
            self._pypsa_network_template.add("Load", f"base_{x}", bus=x, p_set=0)

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
            EV SoC reset to mean; arrival/departure still fixed at 0/95
        """
        rng = _random.Random(seed)
        self._load_multiplier = rng.uniform(const.LOAD_MULTIPLIER_MIN, const.LOAD_MULTIPLIER_MAX)

        self._pypsa_network = self._pypsa_network_template.copy()
        self._current_step = 0
        for x in self._evs.values():
            x.soc = const.EV_INITIAL_SOC_MEAN
        observations: dict[str, Observation] = {}
        for x in self.network.household_bus_ids:
            current = Observation(
            agent_id = x,
            soc_progress = self._evs[x].soc,
            time_urgency = max(0,  (self._evs[x].departure_step - self._current_step) / const.EPISODE_STEPS),
            electricity_price = 1.0,
            base_load_kw = get_base_load_kw(0, self._load_multiplier),
            outdoor_temperature_c = 20.0
            )
            observations[x] = current

        return observations


    def step(self, actions: dict[str, ChargingAction]) -> tuple[dict[str, StepResult], PowerFlowResult]:
        """
        advances the network by one step, applying the given strategy
        implementation details:
            base loads from BDEW H0 profile scaled by per-episode load multiplier
            price set to constant 1.0 for now
            curtailment based on max load of lines and trafo, with multiplier defined in const
            reward calculation:
                reward = -electricity_cost * REWARD_ELECTRICITY_COST_WEIGHT
                reward += REWARD_SOC_COMPLETION_BONUS if soc >= target at departure
                reward += REWARD_SOC_MISS_PENALTY if departure and soc < target

        """
        snapshot = self._pypsa_network.snapshots[self._current_step]

        # 1. Apply charging actions
        for agent_id, action in actions.items():
            power_kw = const.ACTION_TO_KW[action]
            self._pypsa_network.loads.at[f"ev_{agent_id}", "p_set"] = power_kw / 1000  # kW to MW

        # 2. Add base loads (BDEW H0 profile, scaled by per-episode multiplier)
        base_load_kw = get_base_load_kw(self._current_step, self._load_multiplier)
        for bus_id in self.network.household_bus_ids:
            self._pypsa_network.loads.at[f"base_{bus_id}", "p_set"] = base_load_kw / 1000  # kW → MW

        # 3. Run power flow
        self._pypsa_network.lpf(snapshots=snapshot)

        # 4. Check for overloads and apply §14a curtailment
        trafo = self._pypsa_network.transformers.index[0]
        trafo_loading = abs(self._pypsa_network.transformers_t.p0.at[snapshot, trafo]) / \
                        self._pypsa_network.transformers.at[trafo, "s_nom"]

        line_loadings = {}
        for line_id in self._pypsa_network.lines.index:
            s_nom = self._pypsa_network.lines.at[line_id, "s_nom"]
            p0 = abs(self._pypsa_network.lines_t.p0.at[snapshot, line_id])
            line_loadings[line_id] = p0 / s_nom

        curtailment_applied = False
        curtailed_power_kw: dict[str, float] = {}

        overloaded = (
            trafo_loading > const.TRANSFORMER_OVERLOAD_THRESHOLD or
            any(l > const.LINE_OVERLOAD_THRESHOLD for l in line_loadings.values())
        )

        if overloaded:
            curtailment_applied = True
            for agent_id in self._evs:
                original_kw = const.ACTION_TO_KW[actions.get(agent_id, ChargingAction.OFF)]
                capped_kw = min(original_kw, const.MAX_CONTROLLED_POWER_KW)
                curtailed_kw = original_kw - capped_kw
                if curtailed_kw > 0:
                    curtailed_power_kw[agent_id] = curtailed_kw
                self._pypsa_network.loads.at[f"ev_{agent_id}", "p_set"] = capped_kw / 1000
            # re-run power flow with curtailed loads
            self._pypsa_network.lpf(snapshots=snapshot)

        # 5. Update EV SoC
        for agent_id in self._evs:
            actual_kw = self._pypsa_network.loads.at[f"ev_{agent_id}", "p_set"] * 1000
            energy_kwh = actual_kw * const.TIMESTEP_HOURS
            self._evs[agent_id].soc = min(
                1.0,
                self._evs[agent_id].soc + energy_kwh / const.EV_BATTERY_CAPACITY_KWH
            )

        # 6. Compute rewards and build results
        step_results: dict[str, StepResult] = {}
        for agent_id, ev in self._evs.items():
            actual_kw = self._pypsa_network.loads.at[f"ev_{agent_id}", "p_set"] * 1000
            electricity_cost = actual_kw * const.TIMESTEP_HOURS * 1.0  # price placeholder
            reward = -electricity_cost * const.REWARD_ELECTRICITY_COST_WEIGHT

            at_departure = self._current_step == ev.departure_step
            if at_departure:
                if ev.soc >= ev.target_soc:
                    reward += const.REWARD_SOC_COMPLETION_BONUS
                else:
                    reward += const.REWARD_SOC_MISS_PENALTY

            obs = Observation(
                agent_id=agent_id,
                soc_progress=ev.soc / ev.target_soc,
                time_urgency=max(0, (ev.departure_step - self._current_step) / const.EPISODE_STEPS),
                electricity_price=1.0,  # TODO: price scenario
                base_load_kw=base_load_kw,
                outdoor_temperature_c=20.0,
            )

            step_results[agent_id] = StepResult(
                agent_id=agent_id,
                observation=obs,
                reward=reward,
                done=self._current_step >= const.EPISODE_STEPS - 1,
                truncated=False,
                info={"curtailed_kw": curtailed_power_kw.get(agent_id, 0.0)}
            )

        # 7. Build PowerFlowResult
        bus_voltages = {
            bus_id: self._pypsa_network.buses_t.v_mag_pu.at[snapshot, bus_id]
            for bus_id in self._pypsa_network.buses.index
        }

        power_flow_result = PowerFlowResult(
            timestep=self._current_step,
            transformer_loading_pu=trafo_loading,
            line_loadings_pu=line_loadings,
            bus_voltages_pu=bus_voltages,
            curtailment_applied=curtailment_applied,
            curtailed_power_kw=curtailed_power_kw
        )

        self._current_step += 1
        return step_results, power_flow_result


