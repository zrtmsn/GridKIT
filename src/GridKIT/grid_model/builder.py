#grid_model/builder.py
import json
from core import GridNetwork, settings
from core.protocols import NetworkBuilderProtocol

class StubNetworkBuilder(NetworkBuilderProtocol):

    def build(self) -> GridNetwork:
        """
        loads a json file from data containing example network specs 
        and converts it into a GridNetwork as defined in core.models.
        """
        with open(settings.stub_network_path) as f:
            return GridNetwork.model_validate(json.load(f))