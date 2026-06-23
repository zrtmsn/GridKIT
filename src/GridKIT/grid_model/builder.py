#grid_model/builder.py
import json
from pathlib import Path
from core import GridNetwork, settings
from core.protocols import NetworkBuilderProtocol

class StubNetworkBuilder(NetworkBuilderProtocol):
    """
    Loads a GridNetwork from a JSON file. Defaults to the small test stub
    (settings.stub_network_path); pass `path` to load a larger feeder
    (e.g. data/feeder_20.json) or a map_ui-exported OSM network.
    """

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path is not None else settings.stub_network_path

    def build(self) -> GridNetwork:
        with open(self._path) as f:
            return GridNetwork.model_validate(json.load(f))