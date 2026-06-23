# grid_model/test_ev_sampler.py
import numpy as np

import core.constants as const
from grid_model.ev_sampler import sample_evs

AGENTS = ["bus_0", "bus_1", "bus_2", "bus_3", "bus_4"]


def test_sample_returns_one_ev_per_agent():
    evs = sample_evs(AGENTS, np.random.default_rng(0))
    assert set(evs.keys()) == set(AGENTS)


def test_departure_after_arrival_with_min_duration():
    evs = sample_evs(AGENTS, np.random.default_rng(0))
    for ev in evs.values():
        assert ev.departure_step - ev.arrival_step >= const.EV_MIN_CONNECTED_STEPS


def test_steps_within_bounds():
    evs = sample_evs(AGENTS, np.random.default_rng(2))
    for ev in evs.values():
        assert ev.arrival_step >= const.EV_ARRIVAL_STEP_MIN
        assert ev.departure_step <= const.EV_DEPARTURE_STEP_MAX
        assert 0.05 <= ev.soc <= 0.95


def test_agents_are_heterogeneous():
    evs = sample_evs(AGENTS, np.random.default_rng(3))
    arrivals = {ev.arrival_step for ev in evs.values()}
    socs = {round(ev.soc, 4) for ev in evs.values()}
    assert len(arrivals) > 1 or len(socs) > 1


def test_same_seed_reproducible():
    a = sample_evs(AGENTS, np.random.default_rng(9))
    b = sample_evs(AGENTS, np.random.default_rng(9))
    for k in AGENTS:
        assert a[k].arrival_step == b[k].arrival_step
        assert a[k].departure_step == b[k].departure_step
        assert a[k].soc == b[k].soc
