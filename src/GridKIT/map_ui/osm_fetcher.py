\
# ─────────────────────────────────────────────────────────────
# map_ui/osm_fetcher.py
#
# OSM / Overpass ingestion for GridKIT.
#
# Responsibility:
#   - receive map-selected coordinates / bounding box from map_widget.py
#   - call Overpass API
#   - parse OSM nodes/ways/buildings
#   - produce core.models.GridNetwork for grid_model
#
# Important boundary:
#   map_ui does not run power flow and does not implement grid physics.
#   It only builds a first topology candidate and exports a GridNetwork.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Iterable

import requests
from pydantic import BaseModel, Field, model_validator

from core.models import BusModel, GridNetwork, LineModel, TransformerModel
from core.constants import NOMINAL_VOLTAGE_KV


# Conservative LV cable defaults used until grid_model replaces them
# with technology-specific parameters.
DEFAULT_R_OHM_PER_KM = 0.642
DEFAULT_X_OHM_PER_KM = 0.083
DEFAULT_MAX_I_KA = 0.20
DEFAULT_TRAFO_MVA = 0.16

# Building → household classification.
# Most OSM buildings (especially in villages) are tagged only `building=yes`, so a
# whitelist of specific residential subtypes misses the majority. We instead count
# ANY building as a household EXCEPT this blacklist of clearly non-residential
# types. Set OsmFetchConfig.strict_residential=True to fall back to the whitelist.
# Explicit residential subtypes (used by strict mode). Note `building=yes` is NOT
# here — strict deliberately excludes it; inclusive mode keeps it via the blacklist.
RESIDENTIAL_BUILDING_TAGS = {
    "house", "detached", "residential", "apartments", "terrace",
    "semidetached_house", "bungalow", "farm", "dormitory", "cabin", "houseboat",
}
NON_RESIDENTIAL_BUILDING_TAGS = {
    "garage", "garages", "carport", "shed", "roof", "greenhouse", "hut", "cabin_hut",
    "industrial", "commercial", "retail", "warehouse", "supermarket", "kiosk",
    "church", "chapel", "cathedral", "mosque", "synagogue", "temple", "shrine",
    "school", "university", "college", "kindergarten", "hospital", "hotel",
    "public", "civic", "government", "office", "barn", "stable", "cowshed",
    "farm_auxiliary", "service", "construction", "ruins", "container",
    "transformer_tower", "water_tower", "silo", "storage_tank", "tank", "bunker",
    "parking", "hangar", "train_station", "toilets", "bridge", "sports_hall",
}


def is_residential_building(building: str | None, *, strict: bool = False) -> bool:
    """Whether an OSM building tag should count as a household connection point.

    Default (inclusive): any building except the non-residential blacklist —
    critically this keeps `building=yes`, the tag most village houses carry.
    strict: only the explicit residential subtype whitelist.
    """
    if not building:
        return False
    if strict:
        return building in RESIDENTIAL_BUILDING_TAGS
    return building not in NON_RESIDENTIAL_BUILDING_TAGS


class AreaBounds(BaseModel):
    """
    Bounding box selected in the GUI.

    Coordinate order is explicit to avoid the common Overpass/GeoJSON mix-up:
    - north/south are latitudes
    - east/west are longitudes
    """

    south: float = Field(ge=-90, le=90)
    west: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def validate_box(self) -> "AreaBounds":
        if self.south >= self.north:
            raise ValueError("south must be smaller than north")
        if self.west >= self.east:
            raise ValueError("west must be smaller than east")
        return self

    @property
    def center_lat(self) -> float:
        return (self.south + self.north) / 2

    @property
    def center_lon(self) -> float:
        return (self.west + self.east) / 2

    def contains(self, lat: float, lon: float) -> bool:
        """True when a point falls inside the drawn box (edges included)."""
        return self.south <= lat <= self.north and self.west <= lon <= self.east

    @property
    def as_overpass_bbox(self) -> str:
        # Overpass order: south, west, north, east
        return f"{self.south},{self.west},{self.north},{self.east}"

    @property
    def as_geojson_bbox(self) -> list[float]:
        # GeoJSON-ish order: west, south, east, north
        return [self.west, self.south, self.east, self.north]

    def approx_area_km2(self) -> float:
        height_km = haversine_km(self.south, self.center_lon, self.north, self.center_lon)
        width_km = haversine_km(self.center_lat, self.west, self.center_lat, self.east)
        return height_km * width_km


class OsmFetchConfig(BaseModel):
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    # Fallback mirrors tried in order when the primary is overloaded (503/504/timeout).
    overpass_mirrors: tuple[str, ...] = (
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
    )
    max_retries: int = 1                 # full extra rounds over all endpoints
    timeout_seconds: int = 45
    max_area_km2: float = 4.0

    # If OpenStreetMap contains no power=* lines in the selected area,
    # fall back to highway ways. This makes demos robust, but the result is
    # only a topology proxy, not a verified electrical grid.
    allow_highway_fallback: bool = True

    # Household detection: inclusive (any building minus a non-residential
    # blacklist — keeps `building=yes`) vs. strict (residential-subtype whitelist).
    strict_residential: bool = False

    # Buildings are connected to the closest network bus if it is within this
    # radius. Otherwise a household bus is created without a service line.
    household_connection_radius_m: float = 120.0


class MapUiBuildResult(BaseModel):
    """
    Complete output of map_ui.

    grid_network is the object handed over to grid_model.
    The remaining fields are UI/diagnostic metadata for display/export.
    """

    bounds: AreaBounds
    grid_network: GridNetwork
    raw_osm_node_count: int
    raw_osm_way_count: int
    selected_coordinate_count: int
    warnings: list[str] = Field(default_factory=list)

    @property
    def bus_count(self) -> int:
        return len(self.grid_network.buses)

    @property
    def line_count(self) -> int:
        return len(self.grid_network.lines)

    @property
    def transformer_count(self) -> int:
        return len(self.grid_network.transformers)

    @property
    def household_count(self) -> int:
        return self.grid_network.n_households

    def summary_dict(self) -> dict[str, Any]:
        return {
            "network_id": self.grid_network.network_id,
            "bounds": self.bounds.model_dump(),
            "selected_coordinate_count": self.selected_coordinate_count,
            "raw_osm_node_count": self.raw_osm_node_count,
            "raw_osm_way_count": self.raw_osm_way_count,
            "bus_count": self.bus_count,
            "line_count": self.line_count,
            "transformer_count": self.transformer_count,
            "household_count": self.household_count,
            "warnings": self.warnings,
        }

    def save_grid_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.grid_network.model_dump_json(indent=2), encoding="utf-8")
        return out

    def save_summary_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json_dumps_pretty(self.summary_dict()), encoding="utf-8")
        return out


@dataclass(frozen=True)
class OsmNode:
    osm_id: int
    lat: float
    lon: float
    tags: dict[str, Any]


@dataclass(frozen=True)
class OsmWay:
    osm_id: int
    node_ids: list[int]
    tags: dict[str, Any]


def build_grid_network_from_bounds(
    bounds: AreaBounds,
    *,
    area_name: str | None = None,
    network_id: str | None = None,
    config: OsmFetchConfig | None = None,
) -> MapUiBuildResult:
    """
    Main entry point used by the GUI and by tests.

    Input:
        AreaBounds from the map selection.

    Output:
        MapUiBuildResult containing a core.models.GridNetwork.
    """

    config = config or OsmFetchConfig()

    if bounds.approx_area_km2() > config.max_area_km2:
        raise ValueError(
            f"Selected area is about {bounds.approx_area_km2():.2f} km². "
            f"Please select at most {config.max_area_km2:.2f} km² to keep Overpass fast."
        )

    osm_json = fetch_overpass_json(bounds, config=config)
    nodes, ways = parse_osm_elements(osm_json)

    power_ways = [
        way for way in ways.values()
        if way.tags.get("power") in {"line", "minor_line", "cable"}
    ]

    used_highway_fallback = False
    topology_ways = power_ways
    warnings: list[str] = []

    if not topology_ways and config.allow_highway_fallback:
        topology_ways = [
            way for way in ways.values()
            if "highway" in way.tags and way.tags.get("highway") not in {"footway", "cycleway", "path"}
        ]
        used_highway_fallback = True
        warnings.append(
            "No OSM power=line/minor_line/cable ways found. "
            "Used highway ways as a topology fallback. Treat electrical parameters as placeholder values."
        )

    buses: list[BusModel] = []
    lines: list[LineModel] = []
    transformers: list[TransformerModel] = []
    household_bus_ids: list[str] = []

    bus_id_by_osm_node: dict[int, str] = {}
    bus_locations: dict[str, tuple[float, float]] = {}

    def add_bus_for_node(node_id: int, prefix: str = "bus") -> str | None:
        if node_id in bus_id_by_osm_node:
            return bus_id_by_osm_node[node_id]
        node = nodes.get(node_id)
        if node is None:
            return None
        # Overpass' `>;` recursion returns EVERY node of any way that merely clips
        # the box, so a single road can drag in geometry tens of km away. Keep the
        # grid inside the area the user actually drew; the line loop below already
        # skips segments whose endpoints were rejected here.
        if not bounds.contains(node.lat, node.lon):
            return None

        bus_id = f"{prefix}_{node_id}"
        bus_id_by_osm_node[node_id] = bus_id
        bus_locations[bus_id] = (node.lat, node.lon)
        buses.append(
            BusModel(
                bus_id=bus_id,
                v_nom_kv=NOMINAL_VOLTAGE_KV,
                x_coord=node.lon,
                y_coord=node.lat,
            )
        )
        return bus_id

    for way in topology_ways:
        for node_id in way.node_ids:
            add_bus_for_node(node_id)

        for idx, (a, b) in enumerate(pairwise(way.node_ids)):
            from_bus = bus_id_by_osm_node.get(a)
            to_bus = bus_id_by_osm_node.get(b)
            node_a = nodes.get(a)
            node_b = nodes.get(b)

            if not from_bus or not to_bus or not node_a or not node_b or from_bus == to_bus:
                continue

            length_km = max(haversine_km(node_a.lat, node_a.lon, node_b.lat, node_b.lon), 0.001)
            line_id = f"osmway_{way.osm_id}_{idx}"

            lines.append(
                LineModel(
                    line_id=line_id,
                    from_bus=from_bus,
                    to_bus=to_bus,
                    length_km=length_km,
                    r_ohm_per_km=DEFAULT_R_OHM_PER_KM,
                    x_ohm_per_km=DEFAULT_X_OHM_PER_KM,
                    max_i_ka=DEFAULT_MAX_I_KA,
                )
            )

    # Explicit transformer/substation information from OSM if available.
    transformer_candidates = find_transformer_candidates(nodes, ways)
    if buses:
        for i, (lat, lon) in enumerate(transformer_candidates[:1]):
            closest = nearest_bus(lat, lon, bus_locations)
            if closest:
                hv_bus = f"trafo_hv_{i}"
                buses.append(
                    BusModel(
                        bus_id=hv_bus,
                        v_nom_kv=20.0,
                        x_coord=lon,
                        y_coord=lat,
                    )
                )
                transformers.append(
                    TransformerModel(
                        trafo_id=f"trafo_{i}",
                        hv_bus=hv_bus,
                        lv_bus=closest,
                        s_nom_mva=DEFAULT_TRAFO_MVA,
                    )
                )

    if buses and not transformers:
        # Create a synthetic grid head near the middle of the area. NOT at buses[0]:
        # that is whichever node the first OSM way happened to start at, which can
        # sit in a far corner — leaving the ⚡ marker off-screen and, worse, rooting
        # the feeders somewhere unrepresentative. grid_model may replace it with a
        # better transformer placement later.
        anchor_id = nearest_bus(bounds.center_lat, bounds.center_lon, bus_locations) or buses[0].bus_id
        anchor = next(b for b in buses if b.bus_id == anchor_id)
        hv_bus = "trafo_hv_synthetic"
        buses.append(
            BusModel(
                bus_id=hv_bus,
                v_nom_kv=20.0,
                x_coord=anchor.x_coord,
                y_coord=anchor.y_coord,
            )
        )
        transformers.append(
            TransformerModel(
                trafo_id="trafo_synthetic",
                hv_bus=hv_bus,
                lv_bus=anchor.bus_id,
                s_nom_mva=DEFAULT_TRAFO_MVA,
            )
        )
        warnings.append("No transformer found in OSM. Added a synthetic transformer at the centre of the area.")

    # Residential buildings become household connection points.
    residential_centroids = building_centroids(nodes, ways, strict=config.strict_residential)
    for idx, (lat, lon) in enumerate(residential_centroids):
        household_bus = f"household_{idx}"
        buses.append(
            BusModel(
                bus_id=household_bus,
                v_nom_kv=NOMINAL_VOLTAGE_KV,
                x_coord=lon,
                y_coord=lat,
            )
        )
        household_bus_ids.append(household_bus)

        closest = nearest_bus(lat, lon, bus_locations)
        if closest is not None:
            c_lat, c_lon = bus_locations[closest]
            dist_km = haversine_km(lat, lon, c_lat, c_lon)
            if dist_km * 1000 <= config.household_connection_radius_m:
                lines.append(
                    LineModel(
                        line_id=f"service_{idx}",
                        from_bus=closest,
                        to_bus=household_bus,
                        length_km=max(dist_km, 0.001),
                        r_ohm_per_km=DEFAULT_R_OHM_PER_KM,
                        x_ohm_per_km=DEFAULT_X_OHM_PER_KM,
                        max_i_ka=DEFAULT_MAX_I_KA,
                    )
                )

    if not buses:
        warnings.append("No usable topology was found in the selected area. Try a larger or different selection.")

    if used_highway_fallback and not household_bus_ids:
        warnings.append("No residential buildings found. household_bus_ids is empty.")

    network = GridNetwork(
        network_id=network_id or make_network_id(bounds),
        buses=buses,
        lines=deduplicate_lines(lines),
        transformers=transformers,
        area_name=area_name,
        household_bus_ids=household_bus_ids,
    )

    selected_coordinate_count = 4
    return MapUiBuildResult(
        bounds=bounds,
        grid_network=network,
        raw_osm_node_count=len(nodes),
        raw_osm_way_count=len(ways),
        selected_coordinate_count=selected_coordinate_count,
        warnings=warnings,
    )


def fetch_overpass_json(bounds: AreaBounds, *, config: OsmFetchConfig | None = None) -> dict[str, Any]:
    """
    Query Overpass for the selected bounding box.

    The query fetches:
      - power lines/cables and transformers/substations
      - residential buildings for household/agent placement
      - highways as optional fallback topology
    """

    config = config or OsmFetchConfig()
    bbox = bounds.as_overpass_bbox

    query = f"""
    [out:json][timeout:{config.timeout_seconds}];
    (
      way["power"~"^(line|minor_line|cable)$"]({bbox});
      node["power"~"^(transformer|substation)$"]({bbox});
      way["power"~"^(transformer|substation)$"]({bbox});
      way["building"]({bbox});
      way["highway"]({bbox});
    );
    out body;
    >;
    out skel qt;
    """

    # Try the primary endpoint, then mirrors; retry the whole set. Overpass public
    # servers routinely return 429/503/504 when busy, so failover is essential.
    endpoints: list[str] = []
    for url in (config.overpass_url, *config.overpass_mirrors):
        if url not in endpoints:
            endpoints.append(url)

    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        for url in endpoints:
            try:
                response = requests.post(
                    url,
                    data={"data": query},
                    timeout=config.timeout_seconds + 15,
                    headers={"User-Agent": "GridKIT-map-ui/0.1"},
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001 — try the next mirror
                last_error = exc
        if attempt < config.max_retries:
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"Overpass is unavailable right now (tried {len(endpoints)} server(s) × "
        f"{config.max_retries + 1} attempt(s)). This is usually temporary server load — "
        f"try again in a moment or select a smaller area. Last error: {last_error}"
    )


def parse_osm_elements(osm_json: dict[str, Any]) -> tuple[dict[int, OsmNode], dict[int, OsmWay]]:
    nodes: dict[int, OsmNode] = {}
    ways: dict[int, OsmWay] = {}

    for element in osm_json.get("elements", []):
        element_type = element.get("type")
        osm_id = int(element["id"])
        tags = dict(element.get("tags") or {})

        if element_type == "node" and "lat" in element and "lon" in element:
            nodes[osm_id] = OsmNode(osm_id=osm_id, lat=float(element["lat"]), lon=float(element["lon"]), tags=tags)

        elif element_type == "way":
            ways[osm_id] = OsmWay(osm_id=osm_id, node_ids=[int(n) for n in element.get("nodes", [])], tags=tags)

    return nodes, ways


def find_transformer_candidates(nodes: dict[int, OsmNode], ways: dict[int, OsmWay]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []

    for node in nodes.values():
        if node.tags.get("power") in {"transformer", "substation"}:
            result.append((node.lat, node.lon))

    for way in ways.values():
        if way.tags.get("power") in {"transformer", "substation"}:
            coords = [(nodes[n].lat, nodes[n].lon) for n in way.node_ids if n in nodes]
            if coords:
                result.append(centroid(coords))

    return result


def building_centroids(
    nodes: dict[int, OsmNode],
    ways: dict[int, OsmWay],
    *,
    strict: bool = False,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []

    for way in ways.values():
        if not is_residential_building(way.tags.get("building"), strict=strict):
            continue

        coords = [(nodes[n].lat, nodes[n].lon) for n in way.node_ids if n in nodes]
        if len(coords) >= 3:
            result.append(centroid(coords))

    return result


def nearest_bus(
    lat: float,
    lon: float,
    bus_locations: dict[str, tuple[float, float]],
) -> str | None:
    best_bus: str | None = None
    best_distance = float("inf")

    for bus_id, (b_lat, b_lon) in bus_locations.items():
        distance = haversine_km(lat, lon, b_lat, b_lon)
        if distance < best_distance:
            best_distance = distance
            best_bus = bus_id

    return best_bus


def centroid(coords: Iterable[tuple[float, float]]) -> tuple[float, float]:
    coords_list = list(coords)
    return (
        sum(lat for lat, _ in coords_list) / len(coords_list),
        sum(lon for _, lon in coords_list) / len(coords_list),
    )


def pairwise(values: list[int]) -> Iterable[tuple[int, int]]:
    for i in range(len(values) - 1):
        yield values[i], values[i + 1]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)

    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius_km * c


def make_network_id(bounds: AreaBounds) -> str:
    return (
        "osm_"
        f"{bounds.south:.5f}_{bounds.west:.5f}_"
        f"{bounds.north:.5f}_{bounds.east:.5f}"
    ).replace("-", "m").replace(".", "p")


def deduplicate_lines(lines: list[LineModel]) -> list[LineModel]:
    seen: set[tuple[str, str, str]] = set()
    result: list[LineModel] = []

    for line in lines:
        key = tuple(sorted([line.from_bus, line.to_bus]) + [f"{line.length_km:.6f}"])
        if key in seen:
            continue
        seen.add(key)
        result.append(line)

    return result


def json_dumps_pretty(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2)
