# grid_model/environment.py
import numpy as np

from core.protocols import GridEnvProtocol
from core.models import ChargingAction, EVState, GridNetwork, Observation, PowerFlowResult, StepResult
from grid_model.builder import StubNetworkBuilder
from grid_model.network import build_pypsa_network
from grid_model.profiles import price_profile, base_load_profile
from grid_model.ev_sampler import sample_evs
import core.constants as const


class GridEnv(GridEnvProtocol):
    """
    Power grid simulation environment implementing GridEnvProtocol from core.

    Call reset() once after construction (and once per episode).

    Per episode (seeded, reproducible):
      - heterogeneous EVs are sampled (arrival / departure / initial SoC)
      - a day-ahead price curve and a BDEW-H0-like base-load curve are built
    Per step:
      - only connected EVs charge; §14a curtailment dims them proportionally
        down to MIN_GUARANTEED_POWER_KW when the trafo/line is overloaded
      - a nonlinear power flow gives physical bus voltages, which (together with
        whether the device was dimmed) feed back into the next observation as a
        lagged, locally-observable congestion signal.
    """

    def __init__(self, price_scenario: str = "medium"):
        self._price_scenario = price_scenario
        self._network = StubNetworkBuilder().build()
        self._pypsa_network_template = build_pypsa_network(self.network)
        self._evs: dict[str, EVState] = {}   # key = agent_id
        self._current_step = 0
        self._rng = np.random.default_rng()

        # episode time series (filled in reset)
        self._price = np.zeros(const.EPISODE_STEPS)
        self._base_load = np.zeros(const.EPISODE_STEPS)

        # per-household controllable + inflexible loads (added once → no duplicates)
        for bus_id in self.network.household_bus_ids:
            self._pypsa_network_template.add("Load", f"ev_{bus_id}", bus=bus_id, p_set=0)
            self._pypsa_network_template.add("Load", f"base_{bus_id}", bus=bus_id, p_set=0)

    # ── Metadata ─────────────────────────────────────────────
    @property
    def network(self) -> GridNetwork:
        return self._network

    @property
    def agent_ids(self) -> list[str]:
        return list(self.network.household_bus_ids)

    @property
    def current_step(self) -> int:
        return self._current_step

    def day_ahead_prices(self) -> list[float]:
        """Published price forecast for the current episode (one value per step)."""
        return self._price.tolist()

    # ── Episode lifecycle ────────────────────────────────────
    def reset(self, *, seed: int | None = None) -> dict[str, Observation]:
        """Start a new episode: sample EVs and build the price / base-load curves."""
        self._rng = np.random.default_rng(seed)
        self._pypsa_network = self._pypsa_network_template.copy()
        self._current_step = 0

        self._evs = sample_evs(self.agent_ids, self._rng)
        self._price = price_profile(self._rng, self._price_scenario)
        self._base_load = base_load_profile(self._rng)

        observations: dict[str, Observation] = {}
        for agent_id, ev in self._evs.items():
            ev.is_connected = self._is_connected(ev, 0)
            observations[agent_id] = self._build_observation(
                ev, local_voltage_pu=1.0, recent_curtailment_ratio=1.0
            )
        return observations

    def step(self, actions: dict[str, ChargingAction]) -> tuple[dict[str, StepResult], PowerFlowResult]:
        """Advance one 15-minute timestep."""
        t = self._current_step
        snapshot = self._pypsa_network.snapshots[t]

        # 1. Determine availability and apply requested EV loads (disconnected → forced OFF)
        requested_kw: dict[str, float] = {}
        for agent_id, ev in self._evs.items():
            connected = self._is_connected(ev, t)
            ev.is_connected = connected
            req = const.ACTION_TO_KW[actions.get(agent_id, ChargingAction.OFF)] if connected else 0.0
            requested_kw[agent_id] = req
            self._pypsa_network.loads.at[f"ev_{agent_id}", "p_set"] = req / 1000  # kW → MW

        # 2. Inflexible base load (BDEW-H0-like)
        for bus_id in self.network.household_bus_ids:
            self._pypsa_network.loads.at[f"base_{bus_id}", "p_set"] = self._base_load[t] / 1000

        # 3. Power flow → physical voltages + element loadings
        self._run_power_flow(snapshot)
        trafo_loading, line_loadings = self._loadings(snapshot)

        # 4. §14a curtailment: proportional dimming with the minimum-power floor
        delivered_kw = dict(requested_kw)
        curtailed_power_kw: dict[str, float] = {}
        max_loading = max([trafo_loading, *line_loadings.values()])
        curtailment_applied = (
            trafo_loading > const.TRANSFORMER_OVERLOAD_THRESHOLD
            or any(l > const.LINE_OVERLOAD_THRESHOLD for l in line_loadings.values())
        )
        if curtailment_applied and sum(requested_kw.values()) > 0:
            scale = const.TRANSFORMER_OVERLOAD_THRESHOLD / max_loading  # < 1
            for agent_id, req in requested_kw.items():
                if req <= 0:
                    continue
                # equal-treatment dimming, never below the §14a guarantee, never above request
                capped = min(req, max(req * scale, const.MIN_GUARANTEED_POWER_KW))
                delivered_kw[agent_id] = capped
                self._pypsa_network.loads.at[f"ev_{agent_id}", "p_set"] = capped / 1000
                if req - capped > 1e-9:
                    curtailed_power_kw[agent_id] = req - capped
            # re-solve with the dimmed loads so reported voltages/loadings are post-curtailment
            self._run_power_flow(snapshot)
            trafo_loading, line_loadings = self._loadings(snapshot)

        # 5. Update SoC from delivered power; 6/7. rewards + next observation
        step_results: dict[str, StepResult] = {}
        for agent_id, ev in self._evs.items():
            delivered = delivered_kw[agent_id]
            if delivered > 0:
                ev.soc = min(1.0, ev.soc + delivered * const.TIMESTEP_HOURS / ev.battery_capacity_kwh)

            req = requested_kw[agent_id]
            curtail_ratio = (delivered / req) if req > 1e-9 else 1.0
            voltage = self._bus_voltage(snapshot, ev.bus_id)

            reward, info = self._reward(ev, delivered, t, curtailed_power_kw.get(agent_id, 0.0))
            obs = self._build_observation(ev, local_voltage_pu=voltage, recent_curtailment_ratio=curtail_ratio)
            step_results[agent_id] = StepResult(
                agent_id=agent_id,
                observation=obs,
                reward=reward,
                done=t >= const.EPISODE_STEPS - 1,
                truncated=False,
                info=info,
            )

        power_flow_result = PowerFlowResult(
            timestep=t,
            transformer_loading_pu=trafo_loading,
            line_loadings_pu=line_loadings,
            bus_voltages_pu={b: self._bus_voltage(snapshot, b) for b in self._pypsa_network.buses.index},
            curtailment_applied=curtailment_applied,
            curtailed_power_kw=curtailed_power_kw,
        )

        self._current_step += 1
        return step_results, power_flow_result

    # ── Helpers ──────────────────────────────────────────────
    @staticmethod
    def _is_connected(ev: EVState, t: int) -> bool:
        return ev.arrival_step <= t < ev.departure_step

    def _build_observation(self, ev: EVState, *, local_voltage_pu: float, recent_curtailment_ratio: float) -> Observation:
        t = self._current_step
        return Observation(
            agent_id=ev.agent_id,
            soc_progress=ev.soc / ev.target_soc,
            time_urgency=max(0.0, (ev.departure_step - t) / const.EPISODE_STEPS),
            electricity_price=float(self._price[min(t, const.EPISODE_STEPS - 1)]),
            base_load_kw=float(self._base_load[min(t, const.EPISODE_STEPS - 1)]),
            outdoor_temperature_c=20.0,   # TODO: daily temperature profile
            local_voltage_pu=local_voltage_pu,
            recent_curtailment_ratio=recent_curtailment_ratio,
        )

    def _reward(self, ev: EVState, delivered_kw: float, t: int, curtailed_kw: float) -> tuple[float, dict]:
        """
        Purely self-interested reward: pay for energy, get a terminal bonus/penalty
        for (not) meeting the SoC target by departure. §14a curtailment is felt only
        indirectly, as missed energy — never as a direct penalty term. Kept in one
        place so a cooperative reward_mode can be layered on later (scenario 4).
        """
        cost = delivered_kw * const.TIMESTEP_HOURS * float(self._price[min(t, const.EPISODE_STEPS - 1)])
        reward = -cost * const.REWARD_ELECTRICITY_COST_WEIGHT
        info: dict = {"curtailed_kw": curtailed_kw, "delivered_kw": delivered_kw}

        if t == ev.departure_step:
            satisfied = ev.soc >= ev.target_soc
            reward += const.REWARD_SOC_COMPLETION_BONUS if satisfied else const.REWARD_SOC_MISS_PENALTY
            info.update(terminal=True, final_soc=ev.soc, target_soc=ev.target_soc, soc_satisfied=satisfied)
        return reward, info

    def _run_power_flow(self, snapshot) -> None:
        """Nonlinear power flow for physical voltages; fall back to linear on non-convergence."""
        try:
            self._pypsa_network.pf(snapshots=snapshot)
        except Exception:
            self._pypsa_network.lpf(snapshots=snapshot)

    def _loadings(self, snapshot) -> tuple[float, dict[str, float]]:
        trafo = self._pypsa_network.transformers.index[0]
        trafo_loading = abs(self._pypsa_network.transformers_t.p0.at[snapshot, trafo]) / \
            self._pypsa_network.transformers.at[trafo, "s_nom"]
        line_loadings = {
            line_id: abs(self._pypsa_network.lines_t.p0.at[snapshot, line_id]) /
            self._pypsa_network.lines.at[line_id, "s_nom"]
            for line_id in self._pypsa_network.lines.index
        }
        return float(trafo_loading), {k: float(v) for k, v in line_loadings.items()}

    def _bus_voltage(self, snapshot, bus_id: str) -> float:
        try:
            return float(self._pypsa_network.buses_t.v_mag_pu.at[snapshot, bus_id])
        except (KeyError, AttributeError):
            return 1.0
