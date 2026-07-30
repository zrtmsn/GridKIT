# grid_model/test_surrogate.py
import warnings

import pytest

from grid_model.builder import StubNetworkBuilder
from grid_model.network import build_pypsa_network
from grid_model.surrogate import RadialPowerFlow

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def net():
    return StubNetworkBuilder().build()


def _pypsa_solve(net, loads_kw):
    n = build_pypsa_network(net)
    for b in net.household_bus_ids:
        n.add("Load", f"L_{b}", bus=b, p_set=loads_kw[b] / 1000)
    snap = n.snapshots[0]
    n.pf(snapshots=snap)
    trafo = net.transformers[0].trafo_id
    tl = abs(n.transformers_t.p0.at[snap, trafo]) / n.transformers.at[trafo, "s_nom"]
    ll = {l: abs(n.lines_t.p0.at[snap, l]) / n.lines.at[l, "s_nom"] for l in n.lines.index}
    vv = {b: float(n.buses_t.v_mag_pu.at[snap, b]) for b in n.buses.index}
    return float(tl), ll, vv


def test_surrogate_matches_pypsa_within_tolerance(net):
    sur = RadialPowerFlow(net)
    hh = net.household_bus_ids
    patterns = [
        {b: 8.4 for b in hh},                                   # all full
        {b: (7.4 if i < 3 else 0) + 1.0 for i, b in enumerate(hh)},
        {hh[0]: 8.4, **{b: 1.0 for b in hh[1:]}},               # one heavy
        {b: 0.5 for b in hh},                                   # light
    ]
    for p in patterns:
        stl, sll, svv = sur.solve({b: p[b] / 1000 for b in hh})
        ptl, pll, pvv = _pypsa_solve(net, p)
        assert abs(stl - ptl) < 0.02
        for l in pll:
            assert abs(sll[l] - pll[l]) < 0.02
        for b in pvv:
            assert abs(svv[b] - pvv[b]) < 0.02


def test_surrogate_head_line_carries_all_households(net):
    sur = RadialPowerFlow(net)
    hh = net.household_bus_ids
    _, line_loadings, _ = sur.solve({b: 0.01 for b in hh})   # 10 kW each
    # the line leaving the LV busbar must be the most loaded (carries everyone)
    head = "lv_to_0"
    assert line_loadings[head] == max(line_loadings.values())


def test_surrogate_voltage_drops_under_load(net):
    sur = RadialPowerFlow(net)
    hh = net.household_bus_ids
    _, _, v_light = sur.solve({b: 0.0 for b in hh})
    _, _, v_heavy = sur.solve({b: 0.008 for b in hh})
    far_bus = hh[-1]
    assert v_light[far_bus] == pytest.approx(1.0)
    assert v_heavy[far_bus] < v_light[far_bus]


# ── multi-feeder (ding0-style forest) ────────────────────────
def _two_feeder_net():
    from core.models import BusModel, GridNetwork, LineModel, TransformerModel
    buses = [BusModel(bus_id=b) for b in ("hvA", "lvA", "h0", "h1", "hvB", "lvB", "h2", "h3")]
    edges = [("lvA", "h0"), ("h0", "h1"), ("lvB", "h2"), ("h2", "h3")]
    lines = [LineModel(line_id=f"l{i}", from_bus=a, to_bus=b, length_km=0.03,
                       r_ohm_per_km=0.6, x_ohm_per_km=0.08, max_i_ka=0.2)
             for i, (a, b) in enumerate(edges)]
    trafos = [  # feeder A: one 0.1 MVA; feeder B: two 0.1 MVA in parallel → 0.2 MVA
        TransformerModel(trafo_id="tA", hv_bus="hvA", lv_bus="lvA", s_nom_mva=0.1),
        TransformerModel(trafo_id="tB1", hv_bus="hvB", lv_bus="lvB", s_nom_mva=0.1),
        TransformerModel(trafo_id="tB2", hv_bus="hvB", lv_bus="lvB", s_nom_mva=0.1),
    ]
    return GridNetwork(network_id="2f", buses=buses, lines=lines, transformers=trafos,
                       household_bus_ids=["h0", "h1", "h2", "h3"])


def test_multi_feeder_detection_and_capacity():
    sur = RadialPowerFlow(_two_feeder_net())
    assert set(sur.feeder_capacity) == {"lvA", "lvB"}
    assert sur.feeder_capacity["lvA"] == pytest.approx(0.1)
    assert sur.feeder_capacity["lvB"] == pytest.approx(0.2)   # parallel transformers summed
    assert sur.household_feeder == {"h0": "lvA", "h1": "lvA", "h2": "lvB", "h3": "lvB"}


def test_multi_feeder_loadings_are_independent():
    sur = RadialPowerFlow(_two_feeder_net())
    # same load on each feeder, but B has twice the capacity → half the loading
    max_l, _, _ = sur.solve({"h0": 0.04, "h1": 0.04, "h2": 0.04, "h3": 0.04})
    assert sur.last_feeder_loadings["lvA"] == pytest.approx(0.8)
    assert sur.last_feeder_loadings["lvB"] == pytest.approx(0.4)
    assert max_l == pytest.approx(0.8)
