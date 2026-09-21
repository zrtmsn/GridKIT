from __future__ import annotations

import re
from collections import defaultdict, deque
from math import cos, radians, sqrt
from typing import Any


DEFAULT_MAP_CENTER = (49.0069, 8.4037)
ALL_NETWORK_OPTION = "__all_network__"


def format_transformer_option(option: str, network=None) -> str:
    """
    Convert internal selectbox values into user-facing labels.
    """
    if option == ALL_NETWORK_OPTION:
        return "Gesamtes Netz anzeigen"

    if network is not None:
        return transformer_area_display_label(network, option)

    return transformer_display_label(option)


def transformer_area_display_label(network, trafo_id: str) -> str:
    """
    Return a user-friendly label for one selectable transformer area.
    """
    selectable_ids = selectable_transformer_ids(network)
    trafo_id = str(trafo_id)

    try:
        area_number = selectable_ids.index(trafo_id) + 1
    except ValueError:
        area_number = None

    info = parse_transformer_id(trafo_id)
    lv_grid_id = info["lv_grid_id"]

    if area_number is None:
        if lv_grid_id is not None:
            return f"Transformatorbereich – Netz {lv_grid_id}"

        return "Transformatorbereich"

    if lv_grid_id is not None:
        return f"Transformatorbereich {area_number} – Netz {lv_grid_id}"

    return f"Transformatorbereich {area_number}"


def selectable_transformers(network) -> list[Any]:
    """
    Return transformers that should be visible as user-facing selection options.
    """
    selected_by_lv_grid: dict[str, Any] = {}
    transformers_without_lv_grid: list[Any] = []

    for transformer in getattr(network, "transformers", []):
        trafo_id = str(transformer.trafo_id)
        info = parse_transformer_id(trafo_id)

        if info["is_reinforced"]:
            continue

        lv_grid_id = info["lv_grid_id"]

        if lv_grid_id is None:
            transformers_without_lv_grid.append(transformer)
            continue

        if lv_grid_id not in selected_by_lv_grid:
            selected_by_lv_grid[lv_grid_id] = transformer

    return list(selected_by_lv_grid.values()) + transformers_without_lv_grid


def selectable_transformer_ids(network) -> list[str]:
    """
    Return user-facing transformer IDs for dropdown and map selection.
    """
    return [str(transformer.trafo_id) for transformer in selectable_transformers(network)]


def filter_network_by_transformer(network, trafo_id: str):
    """
    Return a copy of the GridNetwork containing only the selected transformer area.

    Household assignment is based on model/source identifiers. A household with
    a different source assignment is not kept only because it is topologically
    reachable through a line.
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

    This combines:
    - topology traversal from the selected transformer's low-voltage side
    - explicit area assignment from source/model IDs

    No geographic distance or coordinate-based household assignment is used.
    """
    selected_transformer = transformer_by_id(network, trafo_id)
    selected_transformer_info = parse_transformer_id(str(selected_transformer.trafo_id))
    selected_lv_grid_id = selected_transformer_info["lv_grid_id"]

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

    for transformer in transformer_group_members(network, trafo_id):
        visited.add(str(transformer.hv_bus))
        visited.add(str(transformer.lv_bus))

    visited.update(bus_ids_for_lv_grid(network, selected_lv_grid_id))
    visited.update(household_ids_for_lv_grid(network, selected_lv_grid_id))

    return visited


def assigned_household_ids_for_transformer(network, trafo_id: str) -> set[str]:
    """
    Return the household IDs assigned to the selected transformer area.

    If the transformer has an LV-grid/source area ID, household assignment is
    based on matching source/model identifiers. This avoids assigning households
    only because they are topologically reachable through a line.
    """
    selected_transformer = transformer_by_id(network, trafo_id)
    selected_info = parse_transformer_id(str(selected_transformer.trafo_id))
    selected_lv_grid_id = selected_info["lv_grid_id"]

    if selected_lv_grid_id is not None:
        matching_households = household_ids_for_lv_grid(network, selected_lv_grid_id)

        if matching_households:
            return matching_households

    reachable_bus_ids = reachable_bus_ids_for_transformer(network, trafo_id)

    return {
        str(bus_id)
        for bus_id in getattr(network, "household_bus_ids", [])
        if str(bus_id) in reachable_bus_ids
    }


def line_adjacency(network) -> dict[str, set[str]]:
    """
    Build an undirected adjacency list from all LineModel connections.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)

    for line in network.lines:
        from_bus = str(line.from_bus)
        to_bus = str(line.to_bus)

        adjacency[from_bus].add(to_bus)
        adjacency[to_bus].add(from_bus)

    return adjacency


def transformer_by_id(network, trafo_id: str):
    """
    Return a transformer object by its trafo_id.
    """
    for transformer in network.transformers:
        if str(transformer.trafo_id) == str(trafo_id):
            return transformer

    raise ValueError(f"Transformator nicht im GridNetwork gefunden: {trafo_id}")


def transformer_group_members(network, trafo_id: str) -> list[Any]:
    """
    Return all transformers that belong to the same user-facing transformer area.
    """
    selected_info = parse_transformer_id(trafo_id)
    selected_lv_grid_id = selected_info["lv_grid_id"]

    selected_transformer = transformer_by_id(network, trafo_id)

    if selected_lv_grid_id is None:
        return [selected_transformer]

    members = []

    for transformer in network.transformers:
        info = parse_transformer_id(str(transformer.trafo_id))

        if info["lv_grid_id"] == selected_lv_grid_id:
            members.append(transformer)

    return members or [selected_transformer]


def transformer_marker_rows(network) -> list[dict[str, Any]]:
    """
    Create display rows for transformer markers and selection metadata.
    """
    rows: list[dict[str, Any]] = []

    for transformer in selectable_transformers(network):
        coords = transformer_display_coordinates(network, transformer)

        if coords is None:
            continue

        lat, lon = coords
        trafo_id = str(transformer.trafo_id)
        trafo_info = parse_transformer_id(trafo_id)

        feeder_bus_ids = reachable_bus_ids_for_transformer(network, trafo_id)
        household_ids = sorted(assigned_household_ids_for_transformer(network, trafo_id))

        summary = device_summary_for_households(network, household_ids)

        rows.append(
            {
                "trafo_id": trafo_id,
                "display_label": transformer_area_display_label(network, trafo_id),
                "lv_grid_id": trafo_info["lv_grid_id"],
                "trafo_number": trafo_info["trafo_number"],
                "is_reinforced": trafo_info["is_reinforced"],
                "role": trafo_info["role"],
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
    max_distance_m: float = 50.0,
) -> str | None:
    """
    Return the closest visible transformer marker to a map click.

    This is only used to identify which marker was clicked in the UI.
    It is not used for household assignment or transformer-area filtering.
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
    """
    Approximate the distance between two latitude/longitude points in metres.
    """
    mean_lat = radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111_320.0 * cos(mean_lat)
    dy = (lat2 - lat1) * 111_320.0

    return sqrt(dx * dx + dy * dy)


def map_center_from_transformers_or_network(network) -> tuple[float, float]:
    """
    Calculate a useful map centre from transformer coordinates or bus coordinates.
    """
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
    """
    Summarise real GridCreator device assignments for the current network.
    """
    household_ids = [str(bus_id) for bus_id in getattr(network, "household_bus_ids", [])]
    return device_summary_for_households(network, household_ids)


def device_summary_for_households(network, household_ids: list[str]) -> dict[str, Any]:
    """
    Summarise real GridCreator device assignments for a selected set of households.
    """
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
            total_battery_kwh += float(read_object_value(devices, "battery_kwh", 0.0) or 0.0)

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
        "total_battery_kwh": round(total_battery_kwh, 3),
        "total_pv_kwp": round(total_pv_kwp, 3),
        "has_gridcreator_device_data": bool(devices_by_bus),
    }


def transformer_display_label(trafo_id: str) -> str:
    """
    Return a readable label for technical transformer IDs.
    """
    info = parse_transformer_id(trafo_id)

    if info["lv_grid_id"] is None:
        return f"Trafo {info['raw_id']}"

    trafo_number_suffix = (
        f" {info['trafo_number']}"
        if info["trafo_number"] is not None
        else ""
    )

    if info["is_reinforced"]:
        return f"LV-Grid {info['lv_grid_id']} – Verstärkungs-Trafo{trafo_number_suffix}"

    return f"LV-Grid {info['lv_grid_id']} – Trafo{trafo_number_suffix}"


def parse_transformer_id(trafo_id: str) -> dict[str, Any]:
    """
    Parse GridCreator transformer IDs into display metadata.
    """
    raw_id = str(trafo_id)
    prefix = "Transformer_lv_grid_"

    if not raw_id.startswith(prefix):
        return {
            "raw_id": raw_id,
            "lv_grid_id": None,
            "trafo_number": None,
            "is_reinforced": False,
            "role": "unknown",
        }

    rest = raw_id[len(prefix):]

    if "_reinforced_" in rest:
        lv_grid_id, trafo_number = rest.split("_reinforced_", 1)

        return {
            "raw_id": raw_id,
            "lv_grid_id": lv_grid_id,
            "trafo_number": trafo_number,
            "is_reinforced": True,
            "role": "reinforced_transformer",
        }

    parts = rest.rsplit("_", 1)

    if len(parts) == 2:
        lv_grid_id, trafo_number = parts
    else:
        lv_grid_id = rest
        trafo_number = None

    return {
        "raw_id": raw_id,
        "lv_grid_id": lv_grid_id,
        "trafo_number": trafo_number,
        "is_reinforced": False,
        "role": "regular_transformer",
    }


def lv_grid_id_from_bus_id(bus_id: str) -> str | None:
    """
    Extract the LV-grid ID from a bus or household ID.

    This uses only the model/source identifier and does not use coordinates.
    """
    bus_id = str(bus_id)

    patterns = [
        r"(?:^|_)lvgd_(\d+)(?=_|$)",
        r"(?:^|_)lv_grid_(\d+)(?=_|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, bus_id)

        if match:
            return match.group(1)

    return None


def household_ids_for_lv_grid(network, lv_grid_id: str | None) -> set[str]:
    """
    Return all household IDs that explicitly belong to the given LV grid.
    """
    if lv_grid_id is None:
        return set()

    return {
        str(bus_id)
        for bus_id in getattr(network, "household_bus_ids", [])
        if lv_grid_id_from_bus_id(str(bus_id)) == str(lv_grid_id)
    }


def bus_ids_for_lv_grid(network, lv_grid_id: str | None) -> set[str]:
    """
    Return all bus IDs that explicitly belong to the given LV grid.
    """
    if lv_grid_id is None:
        return set()

    return {
        str(bus.bus_id)
        for bus in getattr(network, "buses", [])
        if lv_grid_id_from_bus_id(str(bus.bus_id)) == str(lv_grid_id)
    }


def transformer_metadata_for_id(trafo_id: str) -> dict[str, Any]:
    """
    Return readable and technical metadata for one transformer ID.
    """
    info = parse_transformer_id(trafo_id)

    return {
        "trafo_id": str(trafo_id),
        "display_label": transformer_display_label(trafo_id),
        "lv_grid_id": info["lv_grid_id"],
        "trafo_number": info["trafo_number"],
        "is_reinforced": info["is_reinforced"],
        "role": info["role"],
    }


def percent(part: int, total: int) -> float:
    """
    Return a percentage rounded to one decimal place.
    """
    if total <= 0:
        return 0.0

    return round(part / total * 100.0, 1)


def read_object_value(obj: Any, attr_name: str, default: Any = None) -> Any:
    """
    Read a value from either a pydantic/object-like model or a plain dictionary.
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(attr_name, default)

    return getattr(obj, attr_name, default)


def bus_by_id(network) -> dict[str, Any]:
    """
    Return all buses keyed by bus_id.
    """
    return {str(bus.bus_id): bus for bus in network.buses}