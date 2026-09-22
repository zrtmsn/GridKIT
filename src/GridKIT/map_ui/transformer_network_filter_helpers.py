from __future__ import annotations

from collections import defaultdict, deque
from math import cos, radians, sqrt
from typing import Any


DEFAULT_MAP_CENTER = (49.0069, 8.4037)  # Karlsruhe
ALL_NETWORK_OPTION = "__all_network__"

DEFAULT_MARKER_CLICK_DISTANCE_M = 50.0
APPROX_METERS_PER_DEGREE = 111_320.0
COORDINATE_MEAN_DIVISOR = 2.0

REINFORCED_TRANSFORMER_COUNT_THRESHOLD = 1

PERCENT_FACTOR = 100.0
PERCENT_DECIMALS = 1
DEVICE_TOTAL_DECIMALS = 3


def format_transformer_option(option: str) -> str:
    """Convert internal selectbox values into user-facing labels."""
    if option == ALL_NETWORK_OPTION:
        return "Gesamtes Netz anzeigen"

    return transformer_area_display_label(option)


def transformer_area_display_label(trafo_id: str) -> str:
    """
    Return a user-friendly label for one selectable transformer area.

    The original transformer ID is kept so labels stay consistent with the
    dashboard and exported network data.
    """
    return f"Transformatorbereich {trafo_id}"


def selectable_transformers(network) -> list[Any]:
    """
    Return one visible transformer option per physical LV busbar.

    Parallel or reinforced transformers with the same lv_bus belong to the
    same electrical area and are shown as one selectable transformer area.
    """
    seen_lv_bus: dict[str, Any] = {}

    for transformer in getattr(network, "transformers", []):
        lv_bus = str(transformer.lv_bus)
        seen_lv_bus.setdefault(lv_bus, transformer)

    return list(seen_lv_bus.values())


def selectable_transformer_ids(network) -> list[str]:
    """Return user-facing transformer IDs for dropdown and map selection."""
    return [str(transformer.trafo_id) for transformer in selectable_transformers(network)]


def filter_network_by_transformer(network, trafo_id: str):
    """
    Return a GridNetwork copy containing only one transformer area.

    Buses and households are selected by topology, starting from the selected
    transformer's LV bus. Reinforced partner transformers are kept together.
    """
    selected_transformer = transformer_by_id(network, trafo_id)
    selected_transformers = transformer_group_members(network, trafo_id)
    reachable_bus_ids = reachable_bus_ids_for_transformer(network, trafo_id)

    assigned_household_ids = assigned_household_ids_for_transformer(network, trafo_id)
    all_household_ids = {
        str(bus_id)
        for bus_id in getattr(network, "household_bus_ids", [])
    }

    excluded_household_ids = all_household_ids - assigned_household_ids

    filtered_bus_ids = set(reachable_bus_ids)
    filtered_bus_ids.difference_update(excluded_household_ids)
    filtered_bus_ids.update(assigned_household_ids)

    filtered_buses = [
        bus
        for bus in network.buses
        if str(bus.bus_id) in filtered_bus_ids
    ]

    filtered_lines = [
        line
        for line in network.lines
        if str(line.from_bus) in filtered_bus_ids
        and str(line.to_bus) in filtered_bus_ids
    ]

    filtered_household_bus_ids = [
        str(bus_id)
        for bus_id in getattr(network, "household_bus_ids", [])
        if str(bus_id) in assigned_household_ids
    ]

    filtered_household_load_profile_kw = {
        str(bus_id): profile
        for bus_id, profile in getattr(network, "household_load_profile_kw", {}).items()
        if str(bus_id) in assigned_household_ids
    }

    filtered_ev_availability = {
        str(bus_id): availability
        for bus_id, availability in getattr(network, "ev_availability", {}).items()
        if str(bus_id) in assigned_household_ids
    }

    filtered_household_devices = {
        str(bus_id): devices
        for bus_id, devices in getattr(network, "household_devices", {}).items()
        if str(bus_id) in assigned_household_ids
    }

    return network.model_copy(
        deep=True,
        update={
            "network_id": f"{network.network_id}__trafo_{selected_transformer.trafo_id}",
            "buses": filtered_buses,
            "lines": filtered_lines,
            "transformers": selected_transformers,
            "household_bus_ids": filtered_household_bus_ids,
            "household_load_profile_kw": filtered_household_load_profile_kw,
            "ev_availability": filtered_ev_availability,
            "household_devices": filtered_household_devices,
        },
    )


def reachable_bus_ids_for_transformer(network, trafo_id: str) -> set[str]:
    """
    Find buses belonging to the selected transformer area.

    The search follows line topology from the selected LV bus and stops before
    entering buses that belong to other transformer areas.
    """
    selected_transformer = transformer_by_id(network, trafo_id)

    selected_transformer_ids = {
        str(transformer.trafo_id)
        for transformer in transformer_group_members(network, trafo_id)
    }

    start_bus = str(selected_transformer.lv_bus)
    adjacency = line_adjacency(network)

    other_transformer_buses: set[str] = set()

    for transformer in network.transformers:
        if str(transformer.trafo_id) in selected_transformer_ids:
            continue

        other_transformer_buses.add(str(transformer.hv_bus))
        other_transformer_buses.add(str(transformer.lv_bus))

    visited: set[str] = set()
    queue: deque[str] = deque([start_bus])

    while queue:
        current_bus = queue.popleft()

        if current_bus in visited:
            continue

        visited.add(current_bus)

        for neighbour in adjacency.get(current_bus, set()):
            if neighbour in other_transformer_buses:
                continue

            if neighbour not in visited:
                queue.append(neighbour)

    # Keep all buses of reinforced transformers in the selected area.
    for transformer in transformer_group_members(network, trafo_id):
        visited.add(str(transformer.hv_bus))
        visited.add(str(transformer.lv_bus))

    return visited


def assigned_household_ids_for_transformer(network, trafo_id: str) -> set[str]:
    """
    Return households assigned to the selected transformer area.

    Assignment is based on the reachable topology, not on geographic distance.
    """
    reachable_bus_ids = reachable_bus_ids_for_transformer(network, trafo_id)

    return {
        str(bus_id)
        for bus_id in getattr(network, "household_bus_ids", [])
        if str(bus_id) in reachable_bus_ids
    }


def line_adjacency(network) -> dict[str, set[str]]:
    """Build an undirected adjacency list from all line connections."""
    adjacency: dict[str, set[str]] = defaultdict(set)

    for line in network.lines:
        from_bus = str(line.from_bus)
        to_bus = str(line.to_bus)

        adjacency[from_bus].add(to_bus)
        adjacency[to_bus].add(from_bus)

    return adjacency


def transformer_by_id(network, trafo_id: str):
    """Return a transformer object by its trafo_id."""
    for transformer in network.transformers:
        if str(transformer.trafo_id) == str(trafo_id):
            return transformer

    raise ValueError(f"Transformator nicht im GridNetwork gefunden: {trafo_id}")


def transformer_group_members(network, trafo_id: str) -> list[Any]:
    """
    Return all transformers sharing the selected transformer's LV bus.

    This keeps reinforced or parallel transformers together as one area.
    """
    selected_transformer = transformer_by_id(network, trafo_id)
    lv_bus = str(selected_transformer.lv_bus)

    return [transformer for transformer in network.transformers if str(transformer.lv_bus) == lv_bus]


def transformer_marker_rows(network) -> list[dict[str, Any]]:
    """Create display rows for transformer map markers and metadata."""
    rows: list[dict[str, Any]] = []

    for transformer in selectable_transformers(network):
        coords = transformer_display_coordinates(network, transformer)

        if coords is None:
            continue

        lat, lon = coords
        trafo_id = str(transformer.trafo_id)

        feeder_bus_ids = reachable_bus_ids_for_transformer(network, trafo_id)
        household_ids = sorted(assigned_household_ids_for_transformer(network, trafo_id))

        summary = device_summary_for_households(network, household_ids)

        rows.append(
            {
                "trafo_id": trafo_id,
                "display_label": transformer_area_display_label(trafo_id),
                "is_reinforced": (
                    len(transformer_group_members(network, trafo_id))
                    > REINFORCED_TRANSFORMER_COUNT_THRESHOLD
                ),
                "lat": lat,
                "lon": lon,
                "household_count": len(household_ids),
                "bus_count": len(feeder_bus_ids),
                "ev_count": summary["ev_count"],
                "heat_pump_count": summary["heat_pump_count"],
                "battery_count": summary["battery_count"],
                "pv_count": summary["pv_count"],
                "s_nom_mva": transformer.s_nom_mva,
                "vn_hv_kv": transformer.vn_hv_kv,
                "vn_lv_kv": transformer.vn_lv_kv,
            }
        )

    return rows


def transformer_display_coordinates(network, transformer) -> tuple[float, float] | None:
    """
    Return map display coordinates for a transformer.

    The LV bus is preferred. If it has no coordinates, the HV bus is used.
    """
    buses = bus_by_id(network)

    for bus_id in (transformer.lv_bus, transformer.hv_bus):
        bus = buses.get(str(bus_id))

        if bus is None:
            continue

        if bus.x_coord is not None and bus.y_coord is not None:
            return float(bus.y_coord), float(bus.x_coord)

    return None


def nearest_transformer_id(
    network,
    clicked_lat: float,
    clicked_lon: float,
    max_distance_m: float = DEFAULT_MARKER_CLICK_DISTANCE_M,
) -> str | None:
    """
    Return the closest visible transformer marker to a map click.

    This only identifies the clicked marker in the UI. It is not used for
    household assignment or transformer-area filtering.
    """
    nearest_id: str | None = None
    nearest_distance: float | None = None

    for row in transformer_marker_rows(network):
        distance = distance_m(
            clicked_lat,
            clicked_lon,
            row["lat"],
            row["lon"],
        )

        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_id = row["trafo_id"]

    if nearest_distance is None:
        return None

    if nearest_distance > max_distance_m:
        return None

    return nearest_id


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate the distance between two latitude/longitude points in metres."""
    mean_lat = radians((lat1 + lat2) / COORDINATE_MEAN_DIVISOR)
    dx = (lon2 - lon1) * APPROX_METERS_PER_DEGREE * cos(mean_lat)
    dy = (lat2 - lat1) * APPROX_METERS_PER_DEGREE

    return sqrt(dx * dx + dy * dy)


def map_center_from_transformers_or_network(network) -> tuple[float, float]:
    """Calculate a useful map centre from transformer or bus coordinates."""
    transformer_rows = transformer_marker_rows(network)

    if transformer_rows:
        lat = sum(row["lat"] for row in transformer_rows) / len(transformer_rows)
        lon = sum(row["lon"] for row in transformer_rows) / len(transformer_rows)

        return lat, lon

    bus_coords = [
        (float(bus.y_coord), float(bus.x_coord))
        for bus in network.buses
        if bus.x_coord is not None and bus.y_coord is not None
    ]

    if bus_coords:
        lat = sum(coord[0] for coord in bus_coords) / len(bus_coords)
        lon = sum(coord[1] for coord in bus_coords) / len(bus_coords)

        return lat, lon

    return DEFAULT_MAP_CENTER


def device_summary(network) -> dict[str, Any]:
    """Summarise real GridCreator device assignments for the current network."""
    household_ids = [str(bus_id) for bus_id in getattr(network, "household_bus_ids", [])]
    return device_summary_for_households(network, household_ids)


def device_summary_for_households(network, household_ids: list[str]) -> dict[str, Any]:
    """Summarise real GridCreator device assignments for selected households."""
    devices_by_bus = getattr(network, "household_devices", {}) or {}
    household_id_set = {str(bus_id) for bus_id in household_ids}

    ev_count = 0
    heat_pump_count = 0
    battery_count = 0
    pv_count = 0
    total_battery_kwh = 0.0
    total_pv_kwp = 0.0

    for bus_id in household_id_set:
        devices = devices_by_bus.get(str(bus_id))

        if devices is None:
            continue

        has_ev = bool(read_object_value(devices, "ev", False))
        has_heat_pump = bool(read_object_value(devices, "heat_pump", False))
        has_battery = bool(read_object_value(devices, "battery", False))
        has_pv = bool(read_object_value(devices, "pv", False))

        if has_ev:
            ev_count += 1

        if has_heat_pump:
            heat_pump_count += 1

        if has_battery:
            battery_count += 1
            total_battery_kwh += float(
                read_object_value(devices, "battery_kwh", 0.0) or 0.0
            )

        if has_pv:
            pv_count += 1
            total_pv_kwp += float(read_object_value(devices, "pv_kwp", 0.0) or 0.0)

    household_count = len(household_id_set)

    return {
        "household_count": household_count,
        "ev_count": ev_count,
        "heat_pump_count": heat_pump_count,
        "battery_count": battery_count,
        "pv_count": pv_count,
        "ev_share_percent": percent(ev_count, household_count),
        "heat_pump_share_percent": percent(heat_pump_count, household_count),
        "battery_share_percent": percent(battery_count, household_count),
        "pv_share_percent": percent(pv_count, household_count),
        "total_battery_kwh": round(total_battery_kwh, DEVICE_TOTAL_DECIMALS),
        "total_pv_kwp": round(total_pv_kwp, DEVICE_TOTAL_DECIMALS),
        "has_gridcreator_device_data": bool(devices_by_bus),
    }


def transformer_display_label(trafo_id: str) -> str:
    """
    Return a readable label for a transformer ID.

    The ID is already cleaned by grid_model.builder.assign_clean_ids.
    """
    return f"Transformator {trafo_id}"


def transformer_metadata_for_id(trafo_id: str, network) -> dict[str, Any]:
    """Return readable and technical metadata for one transformer ID."""
    return {
        "trafo_id": str(trafo_id),
        "display_label": transformer_display_label(trafo_id),
        "is_reinforced": (
            len(transformer_group_members(network, trafo_id))
            > REINFORCED_TRANSFORMER_COUNT_THRESHOLD
        ),
    }


def percent(part: int, total: int) -> float:
    """Return a percentage rounded to one decimal place."""
    if total <= 0:
        return 0.0

    return round(part / total * PERCENT_FACTOR, PERCENT_DECIMALS)


def read_object_value(obj: Any, attr_name: str, default: Any = None) -> Any:
    """Read a value from either an object-like model or a dictionary."""
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(attr_name, default)

    return getattr(obj, attr_name, default)


def bus_by_id(network) -> dict[str, Any]:
    """Return all buses keyed by bus_id."""
    return {str(bus.bus_id): bus for bus in network.buses}