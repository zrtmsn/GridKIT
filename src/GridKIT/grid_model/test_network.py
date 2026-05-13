#grid_model/test_network.py
import pypsa
import time
from grid_model.network import build_pypsa_network
from .builder import StubNetworkBuilder
from core import settings as st
from core import constants as ct

builder = StubNetworkBuilder()
network = builder.build()
pypsa_network = build_pypsa_network(network)

def test_build_pypsa_network_returns_pypsa_network():
    assert(isinstance(pypsa_network, pypsa.Network))

def test_number_of_buses():
    assert(len(pypsa_network.buses.index) == len(network.buses))

def test_number_of_lines():
    assert(len(pypsa_network.lines.index) == len(network.lines))

def test_number_of_trafos():
    assert(len(pypsa_network.transformers.index) == len(network.transformers))

def test_has_slack_generator():
    assert(len(pypsa_network.generators.index) == 1)

def test_slack_generator_is_on_hv_bus():
    slack = pypsa_network.generators.iloc[0]
    assert slack["bus"] == network.transformers[0].hv_bus

def test_slack_generator_has_correct_control():
    slack = pypsa_network.generators.iloc[0]
    assert slack["control"] == "Slack"

def test_lpf_runs_successfully():
    pypsa_network.lpf()

def test_lpf_performance():
    start = time.time()
    pypsa_network.lpf()
    elapsed = time.time() - start
    print(f"\nlpf() took {elapsed:.3f}s")
    assert elapsed < 1.0  # fail if lpf takes more than 1 second