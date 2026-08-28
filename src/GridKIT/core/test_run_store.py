from core import run_store as rs
from core.models import BusModel, GridNetwork


def _net(n=4):
    buses = [BusModel(bus_id=f"h{i}") for i in range(n)]
    return GridNetwork(network_id="t", buses=buses, household_bus_ids=[f"h{i}" for i in range(n)])


def _config(ev_ids=("h0",), hp_ids=()):
    return {
        "configuration_type": "household_scenario_configuration",
        "resolved": {
            "ev_bus_ids": list(ev_ids),
            "heat_pump_bus_ids": list(hp_ids),
            "load_scaling_by_bus": {},
        },
    }


def test_create_and_load_roundtrip(tmp_path):
    net = _net(6)
    cfg = _config(ev_ids=["h0", "h1"], hp_ids=["h2"])
    rid = rs.create_run("my run", net, cfg, iterations=5, seeds=3, root=tmp_path)

    loaded_cfg = rs.load_config(rid, root=tmp_path)
    assert loaded_cfg["name"] == "my run" and loaded_cfg["iterations"] == 5 and loaded_cfg["n_households"] == 6
    assert rs.get_status(rid, root=tmp_path)["state"] == rs.QUEUED

    net2 = rs.load_network(rid, root=tmp_path)
    assert set(net2.household_bus_ids) == set(net.household_bus_ids)

    cfg2 = rs.load_household_configuration(rid, root=tmp_path)
    assert cfg2["resolved"]["ev_bus_ids"] == ["h0", "h1"]
    assert cfg2["resolved"]["heat_pump_bus_ids"] == ["h2"]


def test_save_network_only_defaults_to_zero_iterations_and_seeds(tmp_path):
    net = _net()
    rid = rs.save_network_only("design only", net, _config(), root=tmp_path)
    cfg = rs.load_config(rid, root=tmp_path)
    assert cfg["iterations"] == 0 and cfg["seeds"] == 0
    assert rs.get_status(rid, root=tmp_path)["state"] == rs.QUEUED


def test_set_status_merges_and_stamps(tmp_path):
    net = _net()
    rid = rs.create_run("r", net, _config(), iterations=10, seeds=2, root=tmp_path)
    rs.set_status(rid, state=rs.RUNNING, progress=0.4, iteration=4, root=tmp_path)
    st = rs.get_status(rid, root=tmp_path)
    assert st["state"] == rs.RUNNING and st["progress"] == 0.4 and st["iteration"] == 4
    assert st["total_iters"] == 10          # preserved from create
    assert "updated" in st


def test_list_runs_newest_first(tmp_path):
    net = _net()
    rs.create_run("a", net, _config(), iterations=1, seeds=1, run_id="run_20240101_000000", root=tmp_path)
    rs.create_run("b", net, _config(), iterations=1, seeds=1, run_id="run_20240102_000000", root=tmp_path)
    runs = rs.list_runs(root=tmp_path)
    assert [r["run_id"] for r in runs] == ["run_20240102_000000", "run_20240101_000000"]
    assert all("state" in r and "has_results" in r for r in runs)


def test_list_runs_empty(tmp_path):
    assert rs.list_runs(root=tmp_path / "nope") == []


def test_load_results_missing_before_save(tmp_path):
    net = _net()
    rid = rs.create_run("r", net, _config(), iterations=1, seeds=1, root=tmp_path)
    summary, timelines = rs.load_results(rid, root=tmp_path)
    assert summary is None and timelines is None
    assert rs.list_runs(root=tmp_path)[0]["has_results"] is False


def test_save_results_persists_and_marks_done(tmp_path):
    net = _net()
    rid = rs.create_run("r", net, _config(), iterations=1, seeds=1, root=tmp_path)
    rs.save_results(rid, [{"scenario": "1"}], {"1": {"transformer_loading": [0.1, 0.2]}}, root=tmp_path)

    summary, timelines = rs.load_results(rid, root=tmp_path)
    assert summary == [{"scenario": "1"}]
    assert timelines == {"1": {"transformer_loading": [0.1, 0.2]}}
    assert rs.get_status(rid, root=tmp_path)["state"] == rs.DONE
    assert rs.list_runs(root=tmp_path)[0]["has_results"] is True


def test_delete_run_removes_everything(tmp_path):
    net = _net()
    rid = rs.create_run("r", net, _config(), iterations=1, seeds=1, root=tmp_path)
    assert rs.run_dir(rid, tmp_path).exists()

    rs.delete_run(rid, root=tmp_path)
    assert not rs.run_dir(rid, tmp_path).exists()
    assert rs.list_runs(root=tmp_path) == []


def test_delete_run_missing_is_a_noop(tmp_path):
    rs.delete_run("does_not_exist", root=tmp_path)  # must not raise
