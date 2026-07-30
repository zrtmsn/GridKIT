# map_ui/test_osm_fetcher.py
import map_ui.osm_fetcher as osm
from map_ui.osm_fetcher import (
    AreaBounds,
    OsmNode,
    OsmWay,
    build_grid_network_from_bounds,
    building_centroids,
    is_residential_building,
)


# ── building → household classification ──────────────────────
def test_inclusive_counts_building_yes_and_subtypes():
    # the fix: `building=yes` (most village houses) counts as a household
    for tag in ("yes", "house", "detached", "apartments", "farm", "bungalow"):
        assert is_residential_building(tag) is True


def test_inclusive_excludes_obvious_non_residential():
    for tag in ("garage", "shed", "church", "school", "industrial", "retail",
                "transformer_tower", "hangar", "barn"):
        assert is_residential_building(tag) is False


def test_none_or_empty_is_not_residential():
    assert is_residential_building(None) is False
    assert is_residential_building("") is False


def test_strict_mode_excludes_building_yes():
    # strict = residential-subtype whitelist only; `yes` no longer qualifies
    assert is_residential_building("yes", strict=True) is False
    assert is_residential_building("house", strict=True) is True
    assert is_residential_building("garage", strict=True) is False


# ── building_centroids honours the classification ────────────
def _square_way(way_id, n0, tags):
    # 4 corner nodes forming a closed-ish footprint
    nodes = {n0 + k: OsmNode(osm_id=n0 + k, lat=49.0 + 0.001 * k, lon=8.0 + 0.001 * k, tags={})
             for k in range(4)}
    way = OsmWay(osm_id=way_id, node_ids=[n0, n0 + 1, n0 + 2, n0 + 3], tags=tags)
    return nodes, way


def test_building_centroids_inclusive_vs_strict_count():
    nodes, w_yes = _square_way(1, 10, {"building": "yes"})
    n2, w_house = _square_way(2, 20, {"building": "house"})
    n3, w_garage = _square_way(3, 30, {"building": "garage"})
    nodes = {**nodes, **n2, **n3}
    ways = {1: w_yes, 2: w_house, 3: w_garage}

    inclusive = building_centroids(nodes, ways)                 # yes + house (garage excluded)
    strict = building_centroids(nodes, ways, strict=True)       # house only
    assert len(inclusive) == 2
    assert len(strict) == 1


# ── the drawn box actually bounds the grid ───────────────────
# Overpass' `>;` recursion returns every node of any way that clips the box, so a
# road crossing the corner used to drag in geometry tens of km away — which then
# became buses, and hosted the synthetic transformer far off-screen.
_BOX = AreaBounds(south=49.098, west=8.703, north=49.112, east=8.720)   # Oberacker-sized


def _sprawling_response() -> dict:
    """One road inside the box, continuing far outside it (as Overpass returns it)."""
    inside = [
        {"type": "node", "id": 1, "lat": 49.1040, "lon": 8.7100},
        {"type": "node", "id": 2, "lat": 49.1050, "lon": 8.7110},
        {"type": "node", "id": 3, "lat": 49.1060, "lon": 8.7120},
    ]
    far_away = [
        {"type": "node", "id": 4, "lat": 48.9394, "lon": 8.8220},   # ~18 km SE
        {"type": "node", "id": 5, "lat": 49.1617, "lon": 8.6058},   # ~10 km NW
    ]
    road = {"type": "way", "id": 100, "nodes": [4, 1, 2, 3, 5], "tags": {"highway": "residential"}}
    return {"elements": [*inside, *far_away, road]}


def _build(monkeypatch, response):
    monkeypatch.setattr(osm, "fetch_overpass_json", lambda bounds, config=None: response)
    return build_grid_network_from_bounds(_BOX, config=osm.OsmFetchConfig())


def test_buses_outside_the_drawn_box_are_dropped(monkeypatch):
    net = _build(monkeypatch, _sprawling_response()).grid_network
    for bus in net.buses:
        assert _BOX.contains(bus.y_coord, bus.x_coord), f"{bus.bus_id} escaped the drawn area"


def test_synthetic_transformer_sits_inside_the_drawn_box(monkeypatch):
    net = _build(monkeypatch, _sprawling_response()).grid_network
    assert len(net.transformers) == 1
    trafo = net.transformers[0]
    for bus_id in (trafo.hv_bus, trafo.lv_bus):
        bus = next(b for b in net.buses if b.bus_id == bus_id)
        # must be visible on the map the user is looking at, not 18 km away
        assert _BOX.contains(bus.y_coord, bus.x_coord)


def test_lines_to_dropped_buses_are_not_emitted(monkeypatch):
    net = _build(monkeypatch, _sprawling_response()).grid_network
    bus_ids = {b.bus_id for b in net.buses}
    for line in net.lines:
        assert line.from_bus in bus_ids and line.to_bus in bus_ids
