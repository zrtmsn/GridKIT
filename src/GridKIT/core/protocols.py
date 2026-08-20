# core/protocols.py
# ─────────────────────────────────────────────────────────────
# Abstract interfaces that define the contract between modules.
# grid_model implements GridEnvProtocol.
# rl_engine depends ONLY on this interface — never on grid_model directly.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import ChargingAction, Observation, StepResult, PowerFlowResult, GridNetwork


class GridEnvProtocol(ABC):
    """
    Contract between grid_model and rl_engine.

    grid_model must implement this class.
    rl_engine must program against this class only.

    One instance = one episode environment.
    """

    # ── Gymnasium-style interface ─────────────────────────────

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> dict[str, Observation]:
        """
        Start a new episode. Sample scenario from distributions.
        Returns initial observations keyed by agent_id.

        Called once at the start of every episode.
        """
        ...

    @abstractmethod
    def step(
        self, actions: dict[str, ChargingAction]
    ) -> tuple[dict[str, StepResult], PowerFlowResult]:
        """
        Advance the simulation by one 15-minute timestep.

        Sequence inside step():
          1. Apply requested charging power per agent
          2. Add inflexible base loads (BDEW H0)
          3. Run power flow (PyPSA)
          4. Apply §14a curtailment if transformer/line overload
          5. Update EV SoC with actual (curtailed) power
          6. Compute rewards
          7. Return StepResult per agent

        Args:
            actions: {agent_id: ChargingAction}

        Returns:
            ({agent_id: StepResult}, PowerFlowResult)  — per-agent results + network-level power flow
        """
        ...

    # ── Metadata ─────────────────────────────────────────────

    @property
    @abstractmethod
    def network(self) -> GridNetwork:
        """The underlying network topology used in this environment."""
        ...

    @property
    @abstractmethod
    def agent_ids(self) -> list[str]:
        """Ordered list of all agent IDs active in this episode."""
        ...

    @property
    @abstractmethod
    def current_step(self) -> int:
        """Current timestep index (0–95)."""
        ...

    def day_ahead_prices(self) -> list[float]:
        """
        Published day-ahead price forecast for the current episode (one value per step).

        Real day-ahead prices are known in advance, so a price-reactive customer can plan
        its charging window from this. Default returns a flat profile; grid_model overrides
        it with the episode's actual price curve.
        """
        raise NotImplementedError


class NetworkBuilderProtocol(ABC):
    """
    Contract for constructing a GridNetwork.

    Phase 1: StubNetworkBuilder (loads pre-built network from file)
    Phase 2: OSMNetworkBuilder (GridCreator + OSM bounding box → GridNetwork)

    grid_model implements this.
    """

    @abstractmethod
    def build(self) -> GridNetwork:
        """
        Build and return a GridNetwork.
        May block while running OSM pipeline or reading CSVs.
        """
        ...