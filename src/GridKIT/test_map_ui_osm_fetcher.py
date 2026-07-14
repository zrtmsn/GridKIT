import pytest
import requests

import map_ui.osm_fetcher as osm_fetcher
from map_ui.osm_fetcher import (
    AreaBounds,
    GridNetworkBuildError,
    OsmFetchConfig,
    build_grid_network_from_bounds,
    fetch_overpass_json,
    parse_osm_elements,
)


def make_valid_fake_osm_json():
    return {
        "elements": [
            {"type": "node", "id": 1, "lat": 49.0000, "lon": 8.4000},
            {"type": "node", "id": 2, "lat": 49.0005, "lon": 8.4005},
            {"type": "node", "id": 3, "lat": 49.0010, "lon": 8.4010},
            {
                "type": "way",
                "id": 100,
                "nodes": [1, 2, 3],
                "tags": {"power": "line"},
            },
            {
                "type": "node",
                "id": 10,
                "lat": 49.0002,
                "lon": 8.4002,
                "tags": {"power": "transformer"},
            },
            {"type": "node", "id": 20, "lat": 49.0001, "lon": 8.4001},
            {"type": "node", "id": 21, "lat": 49.0001, "lon": 8.4002},
            {"type": "node", "id": 22, "lat": 49.0002, "lon": 8.4002},
            {"type": "node", "id": 23, "lat": 49.0002, "lon": 8.4001},
            {
                "type": "way",
                "id": 200,
                "nodes": [20, 21, 22, 23, 20],
                "tags": {"building": "residential"},
            },
        ]
    }


def make_osm_json_without_power_lines():
    return {
        "elements": [
            {
                "type": "node",
                "id": 10,
                "lat": 49.0002,
                "lon": 8.4002,
                "tags": {"power": "transformer"},
            }
        ]
    }


def make_osm_json_without_transformer():
    return {
        "elements": [
            {"type": "node", "id": 1, "lat": 49.0000, "lon": 8.4000},
            {"type": "node", "id": 2, "lat": 49.0005, "lon": 8.4005},
            {
                "type": "way",
                "id": 100,
                "nodes": [1, 2],
                "tags": {"power": "line"},
            },
        ]
    }


def test_area_bounds_valid_box():
    bounds = AreaBounds(
        south=49.0000,
        west=8.3900,
        north=49.0100,
        east=8.4100,
    )

    assert bounds.south == 49.0000
    assert bounds.west == 8.3900
    assert bounds.north == 49.0100
    assert bounds.east == 8.4100
    assert bounds.approx_area_km2() > 0


def test_area_bounds_invalid_box():
    with pytest.raises(ValueError):
        AreaBounds(
            south=49.0100,
            west=8.3900,
            north=49.0000,
            east=8.4100,
        )


def test_parse_osm_elements():
    osm_json = make_valid_fake_osm_json()

    nodes, ways = parse_osm_elements(osm_json)

    assert 1 in nodes
    assert 100 in ways
    assert ways[100].tags["power"] == "line"
    assert ways[200].tags["building"] == "residential"


def test_build_grid_network_from_bounds_success(monkeypatch):
    def fake_fetch_overpass_json(bounds, *, config=None):
        return make_valid_fake_osm_json()

    monkeypatch.setattr(
        osm_fetcher,
        "fetch_overpass_json",
        fake_fetch_overpass_json,
    )

    bounds = AreaBounds(
        south=49.0000,
        west=8.4000,
        north=49.0020,
        east=8.4020,
    )

    result = build_grid_network_from_bounds(
        bounds,
        area_name="test_area",
        config=OsmFetchConfig(max_area_km2=4.0),
    )

    assert result.grid_network.area_name == "test_area"
    assert result.bus_count > 0
    assert result.line_count > 0
    assert result.transformer_count == 1
    assert result.household_count == 1


def test_build_grid_network_fails_without_power_lines(monkeypatch):
    def fake_fetch_overpass_json(bounds, *, config=None):
        return make_osm_json_without_power_lines()

    monkeypatch.setattr(
        osm_fetcher,
        "fetch_overpass_json",
        fake_fetch_overpass_json,
    )

    bounds = AreaBounds(
        south=49.0000,
        west=8.4000,
        north=49.0020,
        east=8.4020,
    )

    with pytest.raises(GridNetworkBuildError) as exc_info:
        build_grid_network_from_bounds(bounds)

    assert "keine OSM-Stromleitungs- oder Kabeldaten" in str(exc_info.value)
    assert "Netzbetreiber" not in str(exc_info.value)


def test_build_grid_network_fails_without_transformer(monkeypatch):
    def fake_fetch_overpass_json(bounds, *, config=None):
        return make_osm_json_without_transformer()

    monkeypatch.setattr(
        osm_fetcher,
        "fetch_overpass_json",
        fake_fetch_overpass_json,
    )

    bounds = AreaBounds(
        south=49.0000,
        west=8.4000,
        north=49.0020,
        east=8.4020,
    )

    with pytest.raises(GridNetworkBuildError) as exc_info:
        build_grid_network_from_bounds(bounds)

    assert "keine Transformator- oder Umspannwerksdaten" in str(exc_info.value)
    assert "Netzbetreiber" not in str(exc_info.value)


def test_fetch_overpass_json_handles_429(monkeypatch):
    class FakeResponse:
        status_code = 429

        def raise_for_status(self):
            raise requests.HTTPError("429 Too Many Requests")

        def json(self):
            return {}

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    bounds = AreaBounds(
        south=49.0000,
        west=8.4000,
        north=49.0020,
        east=8.4020,
    )

    with pytest.raises(GridNetworkBuildError) as exc_info:
        fetch_overpass_json(bounds)

    assert "zu viele Anfragen" in str(exc_info.value)