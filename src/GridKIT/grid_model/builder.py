#grid_model/builder.py
import json
from core import GridNetwork, settings
from core.protocols import NetworkBuilderProtocol

class StubNetworkBuilder(NetworkBuilderProtocol):
    def build(self) -> GridNetwork:
        with open(settings.stub_network_path) as f:
            return GridNetwork.model_validate(json.load(f))