# scripts/test_run_store.py
from core.models import BusModel, GridNetwork, build_device_layout
from scripts import run_store as rs


def _net(n=4):
    buses = [BusModel(bus_id=f"h{i}") for i in range(n)]
    return GridNetwork(network_id="t", buses=buses, household_bus_ids=[f"h{i}" for i in range(n)])


def test_create_and_load_roundtrip(tmp_path):
    net = _net(6)
    layout = build_device_layout(list(net.household_bus_ids), ev=0.5, battery=0.5, heat_pump=0.5, pv=1.0)
    rid = rs.create_run("my run", net, layout, iterations=5, seeds=3, root=tmp_path)

    cfg = rs.load_config(rid, root=tmp_path)
    assert cfg["name"] == "my run" and cfg["iterations"] == 5 and cfg["n_households"] == 6
    assert rs.get_status(rid, root=tmp_path)["state"] == rs.QUEUED

    net2 = rs.load_network(rid, root=tmp_path)
    assert set(net2.household_bus_ids) == set(net.household_bus_ids)
    layout2 = rs.load_layout(rid, root=tmp_path)
    assert set(layout2) == set(layout)
    assert layout2[list(layout)[0]].pv is True   # pv=1.0 → all homes have PV


def test_set_status_merges_and_stamps(tmp_path):
    net = _net()
    layout = build_device_layout(list(net.household_bus_ids))
    rid = rs.create_run("r", net, layout, iterations=10, seeds=2, root=tmp_path)
    rs.set_status(rid, state=rs.RUNNING, progress=0.4, iteration=4, root=tmp_path)
    st = rs.get_status(rid, root=tmp_path)
    assert st["state"] == rs.RUNNING and st["progress"] == 0.4 and st["iteration"] == 4
    assert st["total_iters"] == 10          # preserved from create
    assert "updated" in st


def test_list_runs_newest_first(tmp_path):
    net = _net()
    layout = build_device_layout(list(net.household_bus_ids))
    rs.create_run("a", net, layout, iterations=1, seeds=1, run_id="run_20240101_000000", root=tmp_path)
    rs.create_run("b", net, layout, iterations=1, seeds=1, run_id="run_20240102_000000", root=tmp_path)
    runs = rs.list_runs(root=tmp_path)
    assert [r["run_id"] for r in runs] == ["run_20240102_000000", "run_20240101_000000"]
    assert all("state" in r and "has_results" in r for r in runs)


def test_list_runs_empty(tmp_path):
    assert rs.list_runs(root=tmp_path / "nope") == []
