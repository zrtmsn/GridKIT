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
