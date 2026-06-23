# grid_model/environment.py
import numpy as np

from core.protocols import GridEnvProtocol
from core.models import ChargingAction, EVState, GridNetwork, Observation, PowerFlowResult, StepResult
from grid_model.builder import StubNetworkBuilder
from grid_model.network import build_pypsa_network
from grid_model.surrogate import RadialPowerFlow
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
      - a power-flow backend gives bus voltages + element loadings, which (with
        whether the device was dimmed) feed the next observation as a lagged,
        locally-observable congestion signal.

    Physics backend:
      - use_surrogate=True (default): fast linearized RadialPowerFlow — used for
        training/experiments. Validated against PyPSA (see test_surrogate.py).
      - use_surrogate=False: PyPSA nonlinear pf — slower, for validation.

    EV penetration:
      - the first round(n_households * ev_penetration) household buses get an EV
        (= the agents); the rest contribute base load only.
    """

    def __init__(
        self,
        price_scenario: str = "medium",
        ev_penetration: float = 1.0,
        use_surrogate: bool = True,
        builder=None,
    ):
        self._price_scenario = price_scenario
        self._ev_penetration = ev_penetration
        self._use_surrogate = use_surrogate

        self._network = (builder or StubNetworkBuilder()).build()
        self._surrogate = RadialPowerFlow(self.network)
        # PyPSA template kept for the validation backend (and load-structure tests)
        self._pypsa_network_template = build_pypsa_network(self.network)
        for bus_id in self.network.household_bus_ids:
            self._pypsa_network_template.add("Load", f"ev_{bus_id}", bus=bus_id, p_set=0)
            self._pypsa_network_template.add("Load", f"base_{bus_id}", bus=bus_id, p_set=0)

        # which household buses host an EV (the agents) — spread evenly across the
        # feeder rather than clustered, so penetration reflects load not location
        all_h = list(self.network.household_bus_ids)
        n_ev = max(1, round(self.network.n_households * ev_penetration))
        idx = sorted(set(np.linspace(0, len(all_h) - 1, n_ev).round().astype(int)))
        self._ev_bus_ids = [all_h[i] for i in idx]

        self._evs: dict[str, EVState] = {}   # key = agent_id (EV bus)
        self._current_step = 0
        self._rng = np.random.default_rng()
        self._price = np.zeros(const.EPISODE_STEPS)
        self._base_load = np.zeros(const.EPISODE_STEPS)

    # ── Metadata ─────────────────────────────────────────────
    @property
    def network(self) -> GridNetwork:
        return self._network

    @property
    def agent_ids(self) -> list[str]:
        return list(self._ev_bus_ids)

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def ev_penetration(self) -> float:
        return self._ev_penetration

    def day_ahead_prices(self) -> list[float]:
        """Published price forecast for the current episode (one value per step)."""
        return self._price.tolist()

    @property
    def episode_base_load_kw(self) -> list[float]:
        """Per-step inflexible base load for the current episode (for dashboards)."""
        return self._base_load.tolist()

    # ── Episode lifecycle ────────────────────────────────────
    def reset(self, *, seed: int | None = None) -> dict[str, Observation]:
        """Start a new episode: sample EVs and build the price / base-load curves."""
        self._rng = np.random.default_rng(seed)
        self._pypsa_network = self._pypsa_network_template.copy()
        self._current_step = 0

        self._evs = sample_evs(self._ev_bus_ids, self._rng)
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
        base_kw = float(self._base_load[t])

        # 1. Availability → requested EV power (disconnected agents forced OFF)
        requested_kw: dict[str, float] = {}
        for agent_id, ev in self._evs.items():
            connected = self._is_connected(ev, t)
            ev.is_connected = connected
            requested_kw[agent_id] = (
                const.ACTION_TO_KW[actions.get(agent_id, ChargingAction.OFF)] if connected else 0.0
            )

        # 2/3. Power flow with requested loads
        trafo_loading, line_loadings, voltages = self._solve(requested_kw, base_kw)

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
                capped = min(req, max(req * scale, const.MIN_GUARANTEED_POWER_KW))
                delivered_kw[agent_id] = capped
                if req - capped > 1e-9:
                    curtailed_power_kw[agent_id] = req - capped
            trafo_loading, line_loadings, voltages = self._solve(delivered_kw, base_kw)

        # 5. SoC update; 6/7. rewards + next observation
        step_results: dict[str, StepResult] = {}
        for agent_id, ev in self._evs.items():
            delivered = delivered_kw[agent_id]
            if delivered > 0:
                ev.soc = min(1.0, ev.soc + delivered * const.TIMESTEP_HOURS / ev.battery_capacity_kwh)

            req = requested_kw[agent_id]
            curtail_ratio = (delivered / req) if req > 1e-9 else 1.0
            reward, info = self._reward(ev, delivered, t, curtailed_power_kw.get(agent_id, 0.0))
            obs = self._build_observation(
                ev, local_voltage_pu=voltages.get(ev.bus_id, 1.0), recent_curtailment_ratio=curtail_ratio
            )
            step_results[agent_id] = StepResult(
                agent_id=agent_id, observation=obs, reward=reward,
                done=t >= const.EPISODE_STEPS - 1, truncated=False, info=info,
            )

        power_flow_result = PowerFlowResult(
            timestep=t,
            transformer_loading_pu=trafo_loading,
            line_loadings_pu=line_loadings,
            bus_voltages_pu=voltages,
            curtailment_applied=curtailment_applied,
            curtailed_power_kw=curtailed_power_kw,
        )
        self._current_step += 1
        return step_results, power_flow_result

    # ── Physics backends ─────────────────────────────────────
    def _solve(self, ev_kw: dict[str, float], base_kw: float):
        """Return (trafo_loading_pu, {line: loading_pu}, {bus: voltage_pu})."""
        bus_load_mw = {
            bus_id: (base_kw + ev_kw.get(bus_id, 0.0)) / 1000.0
            for bus_id in self.network.household_bus_ids
        }
        if self._use_surrogate:
            return self._surrogate.solve(bus_load_mw)
        return self._solve_pypsa(bus_load_mw)

    def _solve_pypsa(self, bus_load_mw: dict[str, float]):
        snapshot = self._pypsa_network.snapshots[self._current_step]
        for bus_id in self.network.household_bus_ids:
            self._pypsa_network.loads.at[f"ev_{bus_id}", "p_set"] = 0.0
            self._pypsa_network.loads.at[f"base_{bus_id}", "p_set"] = bus_load_mw[bus_id]
        try:
            self._pypsa_network.pf(snapshots=snapshot)
        except Exception:
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

    # ── Helpers ──────────────────────────────────────────────
    @staticmethod
    def _is_connected(ev: EVState, t: int) -> bool:
        return ev.arrival_step <= t < ev.departure_step

    def _build_observation(self, ev: EVState, *, local_voltage_pu: float, recent_curtailment_ratio: float) -> Observation:
        t = min(self._current_step, const.EPISODE_STEPS - 1)
        return Observation(
            agent_id=ev.agent_id,
            soc_progress=ev.soc / ev.target_soc,
            time_urgency=max(0.0, (ev.departure_step - self._current_step) / const.EPISODE_STEPS),
            electricity_price=float(self._price[t]),
            base_load_kw=float(self._base_load[t]),
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
