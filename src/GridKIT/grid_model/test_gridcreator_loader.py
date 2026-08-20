# grid_model/test_gridcreator_loader.py
from pathlib import Path

import pytest

from grid_model.surrogate import RadialPowerFlow

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_NC = _REPO_ROOT / "vendor" / "GridCreator" / "Test.nc"
_VENDOR = _REPO_ROOT / "vendor" / "GridCreator" / "ding0_grid_generator.py"

needs_nc = pytest.mark.skipif(not _TEST_NC.exists(),
                              reason="GridCreator submodule / Test.nc not present")
needs_vendor = pytest.mark.skipif(not _VENDOR.exists(),
                                  reason="GridCreator submodule not present")


# ── saved .nc ────────────────────────────────────────────────
@needs_nc
def test_import_ding0_grid_topology():
    from grid_model.gridcreator_loader import load_gridcreator_network
    net = load_gridcreator_network(_TEST_NC)
    # realistic multi-transformer village
    assert len(net.transformers) > 1
    assert net.n_households > 500
    # LV bus of each transformer is the lower-voltage side (feeder root)
    for t in net.transformers:
        lv = next(b for b in net.buses if b.bus_id == t.lv_bus)
        hv = next(b for b in net.buses if b.bus_id == t.hv_bus)
        assert lv.v_nom_kv <= hv.v_nom_kv


@needs_nc
def test_imported_grid_forms_multiple_feeders():
    from grid_model.gridcreator_loader import load_gridcreator_network
    sur = RadialPowerFlow(load_gridcreator_network(_TEST_NC))
    # many feeders, each with real (non-trivial) transformer capacity
    assert len(sur.feeder_capacity) >= 5
    assert all(cap > 0 for cap in sur.feeder_capacity.values())
    # every household is assigned to exactly one feeder
    n_house = sum(len(hs) for hs in sur.feeder_households.values())
    assert len(sur.household_feeder) == n_house


# ── bbox → ding0 archive (GridCreator step 1) ────────────────
# A miniature stand-in for the Zenodo archive: two MV/LV transformers of
# DIFFERENT ratings, each rooting a three-household LV feeder. Enough to prove
# we extract real per-transformer capacity rather than one generic trafo.
_TRAFO_MVA = {"t_a": 0.25, "t_b": 0.63}


def _write_fake_ding0_archive(root: Path) -> None:
    import pypsa

    n = pypsa.Network()
    n.add("Bus", "mv", v_nom=20.0, x=7.71, y=48.00)
    for feeder, (t_id, lon0) in enumerate((("t_a", 7.711), ("t_b", 7.715))):
        root_bus = f"lv_root_{feeder}"
        n.add("Bus", root_bus, v_nom=0.4, x=lon0, y=48.001)
        n.add("Transformer", t_id, bus0="mv", bus1=root_bus,   # LV is always bus1 for ding0
              s_nom=_TRAFO_MVA[t_id], x=0.04, r=0.01)
        prev = root_bus
        for h in range(3):
            hb = f"house_{feeder}_{h}"
            n.add("Bus", hb, v_nom=0.4, x=lon0 + 0.0004 * (h + 1), y=48.001)
            n.add("Line", f"l_{feeder}_{h}", bus0=prev, bus1=hb,
                  length=0.05, r=0.01, x=0.004, s_nom=0.2)   # s_nom <= 0.5 or ding0 drops it
            n.add("Load", f"load_{feeder}_{h}", bus=hb, p_set=0.005)
            prev = hb

    topology = root / "grid_district_1" / "topology"
    topology.mkdir(parents=True)
    n.export_to_csv_folder(str(topology))


@pytest.fixture
def fake_archive(tmp_path):
    grids = tmp_path / "grids"
    grids.mkdir()
    _write_fake_ding0_archive(grids)
    return grids


def test_missing_archive_is_reported_not_guessed(tmp_path):
    from grid_model.gridcreator_loader import (
        Ding0DataMissing,
        build_grid_network_from_ding0,
        ding0_archive_available,
    )
    empty = tmp_path / "nothing"
    assert not ding0_archive_available(empty)
    with pytest.raises(Ding0DataMissing, match="zenodo"):
        build_grid_network_from_ding0(47.9, 7.7, 48.1, 7.8, grids_dir=empty)


def test_archive_detected(fake_archive):
    from grid_model.gridcreator_loader import ding0_archive_available
    assert ding0_archive_available(fake_archive)


@needs_vendor
def test_bbox_yields_real_per_transformer_capacity(fake_archive):
    from grid_model.gridcreator_loader import build_grid_network_from_ding0

    net = build_grid_network_from_ding0(47.999, 7.709, 48.002, 7.719, grids_dir=fake_archive)

    # both feeders come back, each keeping its OWN ding0 rating — the whole point
    # of this path versus OSM's single hardcoded 160 kVA transformer
    assert len(net.transformers) == 2
    assert sorted(round(t.s_nom_mva, 3) for t in net.transformers) == [0.25, 0.63]
    assert net.n_households == 6
    # real coordinates survive, so the grid draws at its true place on the map
    assert all(b.x_coord is not None and b.y_coord is not None for b in net.buses)


@needs_vendor
def test_extracted_bbox_grid_is_simulatable(fake_archive):
    from grid_model.gridcreator_loader import build_grid_network_from_ding0

    sur = RadialPowerFlow(build_grid_network_from_ding0(
        47.999, 7.709, 48.002, 7.719, grids_dir=fake_archive))
    assert len(sur.feeder_capacity) == 2
    assert all(cap > 0 for cap in sur.feeder_capacity.values())
    assert sum(len(hs) for hs in sur.feeder_households.values()) == 6


@needs_vendor
def test_area_outside_archive_is_rejected(fake_archive):
    from grid_model.gridcreator_loader import Ding0AreaNotCovered, build_grid_network_from_ding0
    with pytest.raises(Ding0AreaNotCovered):
        build_grid_network_from_ding0(0.0, 0.0, 0.1, 0.1, grids_dir=fake_archive)   # Atlantic
