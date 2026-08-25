# map_ui/test_osm_fetcher.py
# Covers the highway-fallback logic restored when swapping map_ui to main's
# version (main's build_grid_network_from_bounds had no fallback when OSM has
# no power=line/minor_line/cable data in the drawn area — it just failed).
import pytest

import map_ui.osm_fetcher as osm
from map_ui.osm_fetcher import AreaBounds, GridNetworkBuildError, OsmFetchConfig, build_grid_network_from_bounds

_BOUNDS = AreaBounds(south=49.000, west=8.400, north=49.002, east=8.402)

# Every scenario below needs a transformer candidate — build_grid_network_from_bounds
# raises without one, independent of whichever topology source (power lines vs.
# highway fallback) is under test here.
_TRANSFORMER_NODE = {"type": "node", "id": 900, "lat": 49.0009, "lon": 8.4009, "tags": {"power": "substation"}}


def _power_line_response() -> dict:
    return {"elements": [
        {"type": "node", "id": 1, "lat": 49.0000, "lon": 8.4000, "tags": {}},
        {"type": "node", "id": 2, "lat": 49.0010, "lon": 8.4010, "tags": {}},
        {"type": "way", "id": 100, "nodes": [1, 2], "tags": {"power": "line"}},
        _TRANSFORMER_NODE,
    ]}


def _highway_only_response() -> dict:
    return {"elements": [
        {"type": "node", "id": 1, "lat": 49.0000, "lon": 8.4000, "tags": {}},
        {"type": "node", "id": 2, "lat": 49.0010, "lon": 8.4010, "tags": {}},
        {"type": "way", "id": 100, "nodes": [1, 2], "tags": {"highway": "residential"}},
        _TRANSFORMER_NODE,
    ]}


def _footway_only_response() -> dict:
    return {"elements": [
        {"type": "node", "id": 1, "lat": 49.0000, "lon": 8.4000, "tags": {}},
        {"type": "node", "id": 2, "lat": 49.0010, "lon": 8.4010, "tags": {}},
        {"type": "way", "id": 100, "nodes": [1, 2], "tags": {"highway": "footway"}},
        _TRANSFORMER_NODE,
    ]}


def _no_topology_response() -> dict:
    return {"elements": [_TRANSFORMER_NODE]}


def _build(monkeypatch, response, config=None):
    monkeypatch.setattr(osm, "fetch_overpass_json", lambda bounds, config=None: response)
    return build_grid_network_from_bounds(_BOUNDS, config=config or OsmFetchConfig())


# ── highway fallback ─────────────────────────────────────────
def test_uses_power_lines_directly_when_present(monkeypatch):
    result = _build(monkeypatch, _power_line_response())
    assert not any("Highway" in w for w in result.warnings)
    assert result.line_count >= 1


def test_falls_back_to_highway_when_no_power_lines(monkeypatch):
    result = _build(monkeypatch, _highway_only_response())
    assert any("Highway" in w for w in result.warnings)
    assert result.line_count >= 1


def test_fallback_excludes_footways_cycleways_and_paths(monkeypatch):
    # footway is explicitly excluded from the fallback candidates, so with no
    # power lines and only a footway, there's still nothing to build from
    with pytest.raises(GridNetworkBuildError):
        _build(monkeypatch, _footway_only_response())


def test_fallback_disabled_raises_instead_of_using_highways(monkeypatch):
    config = OsmFetchConfig(allow_highway_fallback=False)
    with pytest.raises(GridNetworkBuildError):
        _build(monkeypatch, _highway_only_response(), config=config)


def test_no_topology_at_all_raises(monkeypatch):
    with pytest.raises(GridNetworkBuildError):
        _build(monkeypatch, _no_topology_response())
