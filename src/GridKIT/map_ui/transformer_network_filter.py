from __future__ import annotations

from collections import defaultdict, deque
from math import cos, radians, sqrt
from typing import Any

import folium
import streamlit as st
from streamlit_folium import st_folium


DEFAULT_MAP_CENTER = (49.0069, 8.4037)
ALL_NETWORK_OPTION = "__all_network__"


def show_transformer_selection(full_network):
    """
    Render the transformer selection UI for a generated GridNetwork.

    Initial state:
    - no transformer is selected
    - the complete generated GridNetwork is displayed

    After a transformer is selected:
    - the network is filtered topology-based from the selected transformer's lv_bus
    - only reachable buses, lines, households, load profiles, EV availability entries
      and household device assignments remain

    No geographic radius is used for deciding which households belong to the transformer.
    """
    st.subheader("Trafo-Auswahl")

    if not getattr(full_network, "transformers", None):
        st.warning("Im erzeugten GridNetwork wurde kein Transformator gefunden.")
        st.session_state["selected_trafo_id"] = None
        st.session_state["built_network"] = full_network
        return full_network

    trafo_options = [str(transformer.trafo_id) for transformer in full_network.transformers]

    current_selected = st.session_state.get("selected_trafo_id")
    if current_selected not in trafo_options:
        current_selected = None
        st.session_state["selected_trafo_id"] = None

    marker_rows = transformer_marker_rows(full_network)

    if marker_rows:
        st.caption(
            "Am Anfang wird das gesamte erzeugte GridNetwork angezeigt. "
            "Klicke auf einen Transformator in der Karte oder wähle ihn über die Auswahlbox, "
            "um nur dieses Trafo-Netz anzuzeigen. Reguläre Trafos und Verstärkungs-Trafos "
            "werden in der Auswahl lesbar gekennzeichnet."
        )

        transformer_map = make_transformer_selection_map(
            full_network,
            selected_trafo_id=current_selected,
        )

        map_key_suffix = current_selected if current_selected is not None else "complete_network"

        clicked_data = st_folium(
            transformer_map,
            height=420,
            width=None,
            returned_objects=["last_object_clicked"],
            key=f"transformer_selection_map_{map_key_suffix}",
        )

        clicked_object = clicked_data.get("last_object_clicked") if clicked_data else None

        if clicked_object:
            clicked_lat = clicked_object.get("lat")
            clicked_lon = clicked_object.get("lng")

            if clicked_lat is not None and clicked_lon is not None:
                clicked_trafo_id = nearest_transformer_id(
                    full_network,
                    clicked_lat=float(clicked_lat),
                    clicked_lon=float(clicked_lon),
                    max_distance_m=50.0,
                )

                if clicked_trafo_id and clicked_trafo_id != st.session_state.get("selected_trafo_id"):
                    apply_network_selection(full_network, clicked_trafo_id)
                    st.rerun()
    else:
        st.info(
            "Transformatoren wurden im GridNetwork gefunden, aber ohne Koordinaten. "
            "Die Auswahl ist deshalb nur über die Auswahlbox möglich."
        )

    options = [ALL_NETWORK_OPTION] + trafo_options

    selected_option = current_selected if current_selected is not None else ALL_NETWORK_OPTION
    selected_index = options.index(selected_option)

    selected_option = st.selectbox(
        "Netzkonfiguration auswählen",
        options=options,
        index=selected_index,
        format_func=format_transformer_option,
        help=(
            "Mit „Gesamtes Netz anzeigen“ wird die vollständige erzeugte Netzwerkkonfiguration angezeigt. "
            "Bei Auswahl eines Transformators wird das GridNetwork topologisch vom lv_bus dieses "
            "Transformators aus gefiltert. Die technische Transformer-ID bleibt intern unverändert; "
            "die Auswahl zeigt nur eine lesbarere Bezeichnung."
        ),
    )

    selected_trafo_id = None if selected_option == ALL_NETWORK_OPTION else selected_option

    if selected_trafo_id != st.session_state.get("selected_trafo_id"):
        apply_network_selection(full_network, selected_trafo_id)
        st.rerun()

    if st.session_state.get("selected_trafo_id") is not None:
        if st.button("Zur kompletten Netzkonfiguration zurückkehren"):
            apply_network_selection(full_network, None)
            st.rerun()

    active_trafo_id = st.session_state.get("selected_trafo_id")

    if active_trafo_id is None:
        st.session_state["built_network"] = full_network

        full_device_summary = device_summary(full_network)

        st.info(
            f"**Gesamte Netzwerkkonfiguration aktiv:** Es ist aktuell kein Trafo-Filter gesetzt.\n\n"
            f"Die folgenden Abschnitte zeigen das vollständig erzeugte GridNetwork für den "
            f"ausgewählten Kartenbereich. Dazu gehören alle enthaltenen Busse, Leitungen, "
            f"Haushalte, Transformatoren und die zugehörigen GridCreator-Gerätedaten.\n\n"
            f"Umfang des vollständigen Netzes: "
            f"**{len(full_network.buses)} Busse**, "
            f"**{len(full_network.lines)} Leitungen**, "
            f"**{len(full_network.household_bus_ids)} Haushalte**, "
            f"**{len(full_network.transformers)} Transformatoren**.\n\n"
            f"Aus GridCreator erkannte Ausstattung: "
            f"**{full_device_summary['ev_count']} EV**, "
            f"**{full_device_summary['heat_pump_count']} Wärmepumpen**, "
            f"**{full_device_summary['battery_count']} Batterien**, "
            f"**{full_device_summary['pv_count']} PV-Anlagen**."
        )

        with st.expander("Verfügbare Transformatoren anzeigen"):
            st.json(
                {
                    "mode": "complete_network",
                    "selected_trafo_id": None,
                    "available_transformers": [
                        transformer_metadata_for_id(trafo_id)
                        for trafo_id in trafo_options
                    ],
                    "filtering": "none",
                    "gridcreator_device_summary": full_device_summary,
                }
            )

        return full_network

    filtered_network = filter_network_by_transformer(full_network, active_trafo_id)
    st.session_state["built_network"] = filtered_network

    selected_transformer = transformer_by_id(full_network, active_trafo_id)
    selected_transformer_label = transformer_display_label(selected_transformer.trafo_id)
    selected_transformer_info = parse_transformer_id(selected_transformer.trafo_id)
    selected_transformer_type = (
        "Verstärkungs-Trafo"
        if selected_transformer_info["is_reinforced"]
        else "Regulärer Trafo"
    )
    filtered_device_summary = device_summary(filtered_network)

    st.info(
        f"**Aktiver Trafo-Filter:** **{selected_transformer_label}** wurde ausgewählt.\n\n"
        f"Technische ID: `{selected_transformer.trafo_id}`\n\n"
        f"Typ: **{selected_transformer_type}**.\n\n"
        f"Die folgenden Abschnitte beziehen sich nun ausschließlich auf das ausgewählte "
        f"Trafo-Netz. Die Netzwerkkonfiguration, die Netzvisualisierung, die "
        f"Haushaltskonfiguration sowie der JSON-Export enthalten nur die Busse, Leitungen, "
        f"Haushalte, Transformator-Daten und GridCreator-Gerätedaten, die diesem "
        f"Trafo-Teilnetz innerhalb des ausgewählten Bereichs zugeordnet wurden.\n\n"
        f"Umfang des ausgewählten Trafo-Netzes: "
        f"**{len(filtered_network.buses)} Busse**, "
        f"**{len(filtered_network.lines)} Leitungen**, "
        f"**{len(filtered_network.household_bus_ids)} Haushalte**.\n\n"
        f"Aus GridCreator erkannte Ausstattung in diesem Trafo-Netz: "
        f"**{filtered_device_summary['ev_count']} EV**, "
        f"**{filtered_device_summary['heat_pump_count']} Wärmepumpen**, "
        f"**{filtered_device_summary['battery_count']} Batterien**, "
        f"**{filtered_device_summary['pv_count']} PV-Anlagen**."
    )

    with st.expander("Technische Trafo-Details anzeigen"):
        st.json(
            {
                "mode": "transformer_feeder",
                "trafo_id": selected_transformer.trafo_id,
                "display_label": selected_transformer_label,
                "lv_grid_id": selected_transformer_info["lv_grid_id"],
                "trafo_number": selected_transformer_info["trafo_number"],
                "is_reinforced": selected_transformer_info["is_reinforced"],
                "role": selected_transformer_info["role"],
                "hv_bus": selected_transformer.hv_bus,
                "lv_bus": selected_transformer.lv_bus,
                "s_nom_mva": selected_transformer.s_nom_mva,
                "vn_hv_kv": selected_transformer.vn_hv_kv,
                "vn_lv_kv": selected_transformer.vn_lv_kv,
                "filtering": "topology_based_from_selected_transformer_lv_bus",
                "gridcreator_device_summary": filtered_device_summary,
            }
        )

    return filtered_network


def format_transformer_option(option: str) -> str:
    """
    Convert internal selectbox values into user-facing labels.
    """
    if option == ALL_NETWORK_OPTION:
        return "Gesamtes Netz anzeigen"

    return transformer_display_label(option)


def apply_network_selection(full_network, trafo_id: str | None) -> None:
    """
    Apply either the complete network or a transformer-filtered network.

    Household-level overrides and global target values are reset when the
    selection changes, because old settings may refer to a different set of
    households.

    Example:
    - complete network: 20 households, EV share 35.0 %
    - selected transformer feeder: 6 households, EV share 16.7 %

    The global sliders in map_widget.py must therefore be rebuilt after every
    switch between complete network and transformer feeder.
    """
    previous_trafo_id = st.session_state.get("selected_trafo_id")

    if trafo_id is None:
        selected_network = full_network
    else:
        selected_network = filter_network_by_transformer(full_network, trafo_id)

    st.session_state["selected_trafo_id"] = trafo_id
    st.session_state["built_network"] = selected_network

    if previous_trafo_id != trafo_id:
        # Individual household overrides are scoped to the currently displayed network.
        # They must not be reused after switching to another transformer feeder.
        st.session_state["household_overrides"] = {}

        # Global device targets are also scoped to the currently displayed network.
        # Without resetting this, the target sliders can keep values from the
        # complete network after a transformer has been selected.
        st.session_state["global_device_targets"] = None
        st.session_state["global_device_target_scope"] = None

        st.session_state.setdefault("household_config_version", 0)
        st.session_state["household_config_version"] += 1

        # Force Streamlit to recreate the global sliders with the new baseline.
        st.session_state.setdefault("scenario_config_version", 0)
        st.session_state["scenario_config_version"] += 1


def make_transformer_selection_map(
    full_network,
    selected_trafo_id: str | None = None,
) -> folium.Map:
    """
    Build a small Folium map with clickable transformer markers.
    """
    rows = transformer_marker_rows(full_network)
    center = map_center_from_transformers_or_network(full_network)

    fmap = folium.Map(
        location=center,
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    for row in rows:
        is_selected = selected_trafo_id is not None and row["trafo_id"] == selected_trafo_id

        icon_color = "orange" if is_selected else "blue"
        tooltip = (
            f"Ausgewählt: {row['display_label']}"
            if is_selected
            else f"Auswählen: {row['display_label']}"
        )

        transformer_type = (
            "Verstärkungs-Trafo"
            if row["is_reinforced"]
            else "Regulärer Trafo"
        )
        lv_grid_label = row["lv_grid_id"] if row["lv_grid_id"] is not None else "Unbekannt"
        trafo_number_label = (
            row["trafo_number"]
            if row["trafo_number"] is not None
            else "Unbekannt"
        )

        popup_html = f"""
        <b>{row["display_label"]}</b><br>
        Technische ID: {row["trafo_id"]}<br>
        LV-Grid: {lv_grid_label}<br>
        Trafo-Nummer: {trafo_number_label}<br>
        Typ: {transformer_type}<br>
        Haushalte im Trafo-Netz: {row["household_count"]}<br>
        Busse im Trafo-Netz: {row["bus_count"]}<br>
        EV: {row["ev_count"]}<br>
        Wärmepumpen: {row["heat_pump_count"]}<br>
        Batterien: {row["battery_count"]}<br>
        PV-Anlagen: {row["pv_count"]}<br>
        Scheinleistung: {row["s_nom_mva"]} MVA<br>
        Spannung HV/LV: {row["vn_hv_kv"]} kV / {row["vn_lv_kv"]} kV<br>
        """

        folium.Marker(
            location=[row["lat"], row["lon"]],
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=400),
            icon=folium.Icon(color=icon_color, icon="bolt", prefix="fa"),
        ).add_to(fmap)

    return fmap


def filter_network_by_transformer(network, trafo_id: str):
    """
    Return a copy of the GridNetwork containing only the selected transformer feeder.

    The resulting network contains:
    - the selected transformer
    - all buses reachable from the selected transformer's lv_bus
    - all lines between reachable buses
    - only household_bus_ids inside this reachable component
    - only load profiles for these households
    - only EV availability entries for these households
    - only GridCreator household device assignments for these households

    This keeps the real GridCreator EV, heat-pump, battery and PV information
    consistent with the currently selected transformer feeder.
    """
    selected_transformer = transformer_by_id(network, trafo_id)
    reachable_bus_ids = reachable_bus_ids_for_transformer(network, trafo_id)

    filtered_buses = [
        bus
        for bus in network.buses
        if str(bus.bus_id) in reachable_bus_ids
    ]

    filtered_lines = [
        line
        for line in network.lines
        if str(line.from_bus) in reachable_bus_ids
        and str(line.to_bus) in reachable_bus_ids
    ]

    filtered_household_bus_ids = [
        str(bus_id)
        for bus_id in getattr(network, "household_bus_ids", [])
        if str(bus_id) in reachable_bus_ids
    ]

    filtered_household_load_profile_kw = {
        str(bus_id): profile
        for bus_id, profile in getattr(network, "household_load_profile_kw", {}).items()
        if str(bus_id) in reachable_bus_ids
    }

    filtered_ev_availability = {
        str(bus_id): availability
        for bus_id, availability in getattr(network, "ev_availability", {}).items()
        if str(bus_id) in reachable_bus_ids
    }

    filtered_household_devices = {
        str(bus_id): devices
        for bus_id, devices in getattr(network, "household_devices", {}).items()
        if str(bus_id) in reachable_bus_ids
    }

    return network.model_copy(
        deep=True,
        update={
            "network_id": f"{network.network_id}__trafo_{selected_transformer.trafo_id}",
            "buses": filtered_buses,
            "lines": filtered_lines,
            "transformers": [selected_transformer],
            "household_bus_ids": filtered_household_bus_ids,
            "household_load_profile_kw": filtered_household_load_profile_kw,
            "ev_availability": filtered_ev_availability,
            "household_devices": filtered_household_devices,
        },
    )


def reachable_bus_ids_for_transformer(network, trafo_id: str) -> set[str]:
    """
    Find all buses reachable from the selected transformer's low-voltage side.

    The traversal follows LineModel connections. Other transformer buses are used
    as boundaries so that the selected feeder does not accidentally include a
    neighbouring transformer's feeder.
    """
    selected_transformer = transformer_by_id(network, trafo_id)
    start_bus = str(selected_transformer.lv_bus)

    adjacency = line_adjacency(network)

    other_transformer_buses: set[str] = set()

    for transformer in network.transformers:
        if str(transformer.trafo_id) == str(trafo_id):
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

    # Keep both transformer-side buses so the selected transformer remains valid.
    visited.add(str(selected_transformer.hv_bus))
    visited.add(str(selected_transformer.lv_bus))

    return visited


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


def transformer_marker_rows(network) -> list[dict[str, Any]]:
    """
    Create display rows for transformer markers and selection metadata.
    """
    rows: list[dict[str, Any]] = []

    for transformer in network.transformers:
        coords = transformer_display_coordinates(network, transformer)

        if coords is None:
            continue

        lat, lon = coords
        trafo_id = str(transformer.trafo_id)
        trafo_info = parse_transformer_id(trafo_id)

        feeder_bus_ids = reachable_bus_ids_for_transformer(network, trafo_id)

        household_ids = [
            str(bus_id)
            for bus_id in getattr(network, "household_bus_ids", [])
            if str(bus_id) in feeder_bus_ids
        ]

        summary = device_summary_for_households(network, household_ids)

        rows.append(
            {
                "trafo_id": trafo_id,
                "display_label": transformer_display_label(trafo_id),
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

    Preference:
    1. low-voltage bus coordinates
    2. high-voltage bus coordinates

    The low-voltage side is preferred because it feeds the household feeder.
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
    Return the closest transformer marker to a map click.

    streamlit-folium gives the clicked coordinates, not always the marker ID.
    Therefore, the click is matched to the nearest transformer marker within a
    small distance threshold. This threshold is only for click matching, not for
    selecting households.
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

        has_ev = bool(getattr(devices, "ev", False))
        has_heat_pump = bool(getattr(devices, "heat_pump", False))
        has_battery = bool(getattr(devices, "battery", False))
        has_pv = bool(getattr(devices, "pv", False))

        if has_ev:
            ev_count += 1

        if has_heat_pump:
            heat_pump_count += 1

        if has_battery:
            battery_count += 1
            total_battery_kwh += float(getattr(devices, "battery_kwh", 0.0) or 0.0)

        if has_pv:
            pv_count += 1
            total_pv_kwp += float(getattr(devices, "pv_kwp", 0.0) or 0.0)

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
    Return a user-friendly label for GridCreator transformer IDs.

    Examples:
    - Transformer_lv_grid_5690200057_1
      -> LV-Grid 5690200057 – Trafo 1

    - Transformer_lv_grid_5690200057_reinforced_3
      -> LV-Grid 5690200057 – Verstärkungs-Trafo 3
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

    Supported patterns:
    - Transformer_lv_grid_<grid_id>_<number>
    - Transformer_lv_grid_<grid_id>_reinforced_<number>
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


def bus_by_id(network) -> dict[str, Any]:
    """
    Return all buses keyed by bus_id.
    """
    return {str(bus.bus_id): bus for bus in network.buses}