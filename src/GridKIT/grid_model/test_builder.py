# grid_model/test_builder.py
from .builder import StubNetworkBuilder
from core.models import GridNetwork

def test_stub_builder_returns_grid_network():
    builder = StubNetworkBuilder()
    network = builder.build()
    assert isinstance(network, GridNetwork)

def test_stub_network_has_households():
    builder = StubNetworkBuilder()
    network = builder.build()
    assert network.n_households == 5

def test_stub_network_has_buses():
    builder = StubNetworkBuilder()
    network = builder.build()
    assert len(network.buses) == 7  # bus_hv, bus_lv, bus_0..4

def test_stub_network_has_lines():
    builder = StubNetworkBuilder()
    network = builder.build()
    assert len(network.lines) == 5

def test_stub_network_has_transformer():
    builder = StubNetworkBuilder()
    network = builder.build()
    assert len(network.transformers) == 1