# grid_model/environment.py
from core.protocols import GridEnvProtocol
from core.models import ChargingAction, EVState, GridNetwork, Observation, PowerFlowResult, StepResult
from grid_model.builder import StubNetworkBuilder
from grid_model.network import build_pypsa_network
import core.constants as const
from core.config import Settings


class GridEnv(GridEnvProtocol):
    
    def __init__(self):
        self._network = StubNetworkBuilder.build()
        self._pypsa_network_template = build_pypsa_network(self.network)
        self._pypsa_network = self._pypsa_network_template.copy()
        self._evs = list[EVState]
        for i in range(len(self.network.household_bus_ids)):
            current_ev = EVState
            current_ev.agent_id, current_ev.bus_id = self.network.household_bus_ids[i]
            self._evs[i] = current_ev
            

        self._current_step = 0
        self._ev_states = {}

    @property
    def network(self) -> GridNetwork:
        return self._network

    @property
    def agent_ids(self) -> list[str]:
        return list(self.network.household_bus_ids)

    @property
    def current_step(self) -> int:
        return self._current_step
    
    def reset(self, seed: int):
        output = dict[str, Observation]
        for x in self.self.network.household_bus_ids:
            current = Observation
            current.agent_id = x

        return output
