# grid_model/test_builder.py
import pytest
from .builder import StubNetworkBuilder
from core.models import GridNetwork

@pytest.fixture
def network():
    return StubNetworkBuilder().build()

def test_stub_builder_returns_grid_network(network):
    assert isinstance(network, GridNetwork)

def test_stub_network_has_households(network):
    assert network.n_households > 0

def test_stub_network_has_buses(network):
    assert len(network.buses) > 0

def test_stub_network_has_lines(network):
    assert len(network.lines) > 0

def test_stub_network_has_transformer(network):
    assert len(network.transformers) > 0