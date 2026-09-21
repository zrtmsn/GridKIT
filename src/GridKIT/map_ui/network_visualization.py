# ─────────────────────────────────────────────────────────────
# map_ui/network_visualization.py
#
# Functions for visualizing a generated GridNetwork on a Folium map.
# This module contains the network map, line rendering, bus rendering,
# and popup content including household scenario values.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from typing import Any

import folium
import streamlit as st
from streamlit_folium import st_folium

from map_ui.household_config import (
    build_household_configuration,
    default_scenario_assumptions,
)
from map_ui.area_bounds import AreaBounds
from map_ui.transformer_network_filter import (
    parse_transformer_id,
    reachable_bus_ids_for_transformer,
    selectable_transformer_ids,
    transformer_area_display_label,
)


def show_network_visualization(network, selected_bounds: AreaBounds) -> None:
    st.subheader("Netzansicht")

    st.markdown(
        """
        **Legende**
        - 🟠 Transformator-Bus
        - 🟢 Haushalt
        - 🔵 Sonstiger Netzknoten
        - Türkise Linien = Leitungen
        """
    )

    household_configuration = build_current_household_configuration_from_session(
        network=network,
        selected_bounds=selected_bounds,
    )

    fmap = create_network_map(
        network=network,
        selected_bounds=selected_bounds,
        household_configuration=household_configuration,
    )

    network_map_key = (
        f"network_visualization_"
        f"{getattr(network, 'network_id', 'network')}_"
        f"{st.session_state.get('scenario_config_version', 0)}_"
        f"{st.session_state.get('household_config_version', 0)}"
    )

    st_folium(
        fmap,
        height=700,
        width=None,
        returned_objects=[],
        key=network_map_key,
    )


def build_current_household_configuration_from_session(
    network,
    selected_bounds: AreaBounds,
) -> dict[str, Any] | None:
    household_ids = sorted(str(bus_id) for bus_id in getattr(network, "household_bus_ids", []))

    if not household_ids:
        return None

    saved_assumptions = st.session_state.get(
        "scenario_assumptions",
        default_scenario_assumptions(),
    )

    household_overrides = st.session_state.get("household_overrides", {})

    return build_household_configuration(
        network=network,
        selected_bounds=selected_bounds,
        household_ids=household_ids,
        ev_share_percent=int(saved_assumptions["ev_share_percent"]),
        heat_pump_share_percent=int(saved_assumptions["heat_pump_share_percent"]),
        global_load_scaling_factor=float(saved_assumptions["global_load_scaling_factor"]),
        selection_seed=int(saved_assumptions["selection_seed"]),
        household_overrides=household_overrides,
    )


def get_bus_scenario_values(
    bus_id: str,
    household_configuration: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if household_configuration is None:
        return None

    resolved = household_configuration.get("resolved", {})

    ev_bus_ids = set(resolved.get("ev_bus_ids", []))
    heat_pump_bus_ids = set(resolved.get("heat_pump_bus_ids", []))
    battery_bus_ids = set(resolved.get("battery_bus_ids", []))
    pv_bus_ids = set(resolved.get("pv_bus_ids", []))
    load_scaling_by_bus = resolved.get("load_scaling_by_bus", {})

    if bus_id not in load_scaling_by_bus:
        return None

    individual_adjustments = household_configuration.get(
        "individual_household_adjustments",
        household_configuration.get("low_level_overrides", {}),
    )

    return {
        "has_ev": bus_id in ev_bus_ids,
        "has_heat_pump": bus_id in heat_pump_bus_ids,
        "has_battery": bus_id in battery_bus_ids,
        "has_pv": bus_id in pv_bus_ids,
        "load_scaling_factor": float(load_scaling_by_bus[bus_id]),
        "is_individual_adjustment": bus_id in individual_adjustments,
    }


def format_bool_de(value: bool) -> str:
    return "Ja" if value else "Nein"


def create_network_map(
    network,
    selected_bounds: AreaBounds,
    household_configuration: dict[str, Any] | None = None,
) -> folium.Map:
    center = (
        (selected_bounds.south + selected_bounds.north) / 2,
        (selected_bounds.west + selected_bounds.east) / 2,
    )

    fmap = folium.Map(
        location=center,
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    folium.Rectangle(
        bounds=[
            [selected_bounds.south, selected_bounds.west],
            [selected_bounds.north, selected_bounds.east],
        ],
        color="#444444",
        weight=2,
        fill=False,
        dash_array="5, 5",
        tooltip="Ausgewählte Bounding Box",
    ).add_to(fmap)

    bus_lookup = {str(bus.bus_id): bus for bus in network.buses}
    household_ids = {str(bus_id) for bus_id in network.household_bus_ids}
    household_label_map = build_household_label_map(network)

    transformer_bus_labels = build_transformer_bus_label_map(network)

    transformer_bus_ids = set(transformer_bus_labels.keys())

    bus_label_map = build_bus_label_map(
        network=network,
        household_label_map=household_label_map,
        transformer_bus_labels=transformer_bus_labels,
    )

    line_group = folium.FeatureGroup(name="Leitungen", show=True)
    household_group = folium.FeatureGroup(name="Haushalte", show=True)
    transformer_group = folium.FeatureGroup(name="Transformator-Busse", show=True)
    bus_group = folium.FeatureGroup(name="Sonstige Netzknoten", show=True)

    add_network_lines(
        line_group=line_group,
        network=network,
        bus_lookup=bus_lookup,
        bus_label_map=bus_label_map,
    )

    add_network_buses(
        household_group=household_group,
        transformer_group=transformer_group,
        bus_group=bus_group,
        network=network,
        household_ids=household_ids,
        household_label_map=household_label_map,
        transformer_bus_ids=transformer_bus_ids,
        transformer_bus_labels=transformer_bus_labels,
        bus_label_map=bus_label_map,
        household_configuration=household_configuration,
    )

    line_group.add_to(fmap)
    household_group.add_to(fmap)
    transformer_group.add_to(fmap)
    bus_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    fmap.fit_bounds(
        [
            [selected_bounds.south, selected_bounds.west],
            [selected_bounds.north, selected_bounds.east],
        ]
    )

    return fmap


def add_network_lines(
    line_group,
    network,
    bus_lookup: dict[str, Any],
    bus_label_map: dict[str, str],
) -> None:
    for line in network.lines:
        from_bus = bus_lookup.get(str(line.from_bus))
        to_bus = bus_lookup.get(str(line.to_bus))

        if from_bus is None or to_bus is None:
            continue

        if (
            from_bus.x_coord is None
            or from_bus.y_coord is None
            or to_bus.x_coord is None
            or to_bus.y_coord is None
        ):
            continue

        from_bus_id = str(line.from_bus)
        to_bus_id = str(line.to_bus)

        from_label = bus_label_map.get(from_bus_id, from_bus_id)
        to_label = bus_label_map.get(to_bus_id, to_bus_id)

        popup_html = f"""
        <b>Leitung:</b> {from_label} → {to_label}<br>
        <b>Technische Leitungs-ID:</b> {line.line_id}<br>
        <b>Von:</b> {from_label}<br>
        <b>Nach:</b> {to_label}<br>
        <b>Technische Start-ID:</b> {line.from_bus}<br>
        <b>Technische Ziel-ID:</b> {line.to_bus}<br>
        <b>Länge (km):</b> {line.length_km:.4f}<br>
        <b>R (Ohm/km):</b> {line.r_ohm_per_km:.4f}<br>
        <b>X (Ohm/km):</b> {line.x_ohm_per_km:.4f}<br>
        <b>Max. Strom (kA):</b> {line.max_i_ka:.4f}
        """

        folium.PolyLine(
            locations=[
                [from_bus.y_coord, from_bus.x_coord],
                [to_bus.y_coord, to_bus.x_coord],
            ],
            color="#0f9d8a",
            weight=3,
            opacity=0.85,
            tooltip=f"Leitung: {from_label} → {to_label}",
            popup=folium.Popup(popup_html, max_width=450),
        ).add_to(line_group)


def add_network_buses(
    household_group,
    transformer_group,
    bus_group,
    network,
    household_ids: set[str],
    household_label_map: dict[str, str],
    transformer_bus_ids: set[str],
    transformer_bus_labels: dict[str, list[str]],
    bus_label_map: dict[str, str],
    household_configuration: dict[str, Any] | None = None,
) -> None:
    for bus in network.buses:
        if bus.x_coord is None or bus.y_coord is None:
            continue

        bus_id = str(bus.bus_id)
        bus_label = bus_label_map.get(bus_id, bus_id)

        popup_html = make_bus_popup(
            bus=bus,
            bus_label=bus_label,
            household_ids=household_ids,
            transformer_bus_ids=transformer_bus_ids,
            transformer_bus_labels=transformer_bus_labels,
            network=network,
            household_configuration=household_configuration,
        )

        if bus_id in transformer_bus_ids:
            folium.CircleMarker(
                location=[bus.y_coord, bus.x_coord],
                radius=8,
                color="#d94801",
                fill=True,
                fill_color="#f16913",
                fill_opacity=0.95,
                weight=2,
                tooltip=f"Transformator-Bus: {bus_label}",
                popup=folium.Popup(popup_html, max_width=450),
            ).add_to(transformer_group)

        elif bus_id in household_ids:
            folium.CircleMarker(
                location=[bus.y_coord, bus.x_coord],
                radius=6,
                color="#238b45",
                fill=True,
                fill_color="#41ab5d",
                fill_opacity=0.9,
                weight=1,
                tooltip=bus_label,
                popup=folium.Popup(popup_html, max_width=450),
            ).add_to(household_group)

        else:
            folium.CircleMarker(
                location=[bus.y_coord, bus.x_coord],
                radius=5,
                color="#2171b5",
                fill=True,
                fill_color="#4292c6",
                fill_opacity=0.85,
                weight=1,
                tooltip=bus_label,
                popup=folium.Popup(popup_html, max_width=450),
            ).add_to(bus_group)


def make_bus_popup(
    bus,
    bus_label: str,
    household_ids: set[str],
    transformer_bus_ids: set[str],
    transformer_bus_labels: dict[str, list[str]],
    network,
    household_configuration: dict[str, Any] | None = None,
) -> str:
    bus_id = str(bus.bus_id)
    roles: list[str] = []

    if bus_id in transformer_bus_ids:
        roles.append("Transformator-Bus")
    if bus_id in household_ids:
        roles.append("Haushalt")
    if not roles:
        roles.append("Sonstiger Netzknoten")

    has_ev_from_gridnetwork = bus_id in getattr(network, "ev_availability", {})
    has_load_profile = bus_id in getattr(network, "household_load_profile_kw", {})

    scenario_values = get_bus_scenario_values(
        bus_id=bus_id,
        household_configuration=household_configuration,
    )

    transformer_area_html = ""

    if bus_id in transformer_bus_labels:
        labels = transformer_bus_labels.get(bus_id, [])
        if labels:
            transformer_area_html = (
                "<br>"
                f"<b>Zugehöriger Transformatorbereich:</b> {', '.join(labels)}"
            )

    scenario_html = ""

    if scenario_values is not None:
        scenario_html = f"""
        <br>
        <b>EV im aktuellen Szenario:</b> {format_bool_de(scenario_values["has_ev"])}<br>
        <b>WP im aktuellen Szenario:</b> {format_bool_de(scenario_values["has_heat_pump"])}<br>
        <b>Batterie im aktuellen Szenario:</b> {format_bool_de(scenario_values["has_battery"])}<br>
        <b>PV im aktuellen Szenario:</b> {format_bool_de(scenario_values["has_pv"])}<br>
        <b>Lastprofil-Skalierung:</b> {scenario_values["load_scaling_factor"]:.2f}<br>
        <b>Individuell angepasst:</b> {format_bool_de(scenario_values["is_individual_adjustment"])}
        """

    return f"""
    <b>Bezeichnung:</b> {bus_label}<br>
    <b>Rolle:</b> {", ".join(roles)}
    {transformer_area_html}
    <br>
    <b>Technische Bus-ID:</b> {bus.bus_id}<br>
    <b>Nominalspannung (kV):</b> {bus.v_nom_kv}<br>
    <b>x_coord:</b> {bus.x_coord}<br>
    <b>y_coord:</b> {bus.y_coord}<br>
    <b>Lastprofil vorhanden:</b> {"Ja" if has_load_profile else "Nein"}<br>
    <b>EV-Verfügbarkeit vorhanden:</b> {"Ja" if has_ev_from_gridnetwork else "Nein"}
    {scenario_html}
    """


def build_household_label_map(network) -> dict[str, str]:
    raw_household_ids = sorted(str(bus_id) for bus_id in getattr(network, "household_bus_ids", []))
    reference_household_ids = household_label_reference_order(network)
    household_ids = sort_household_ids_for_display(raw_household_ids, reference_household_ids)

    return {
        household_id: household_display_label(reference_household_ids, household_id)
        for household_id in household_ids
    }


def household_label_reference_order(network) -> list[str]:
    """
    Return household IDs in a stable, user-friendly order.

    The numbering follows the order of the user-facing transformer areas.
    If the complete generated network is available in session_state, it is used
    so numbering stays stable even when a transformer filter is active.
    """
    full_network = st.session_state.get("full_network") or network

    all_household_ids = {
        str(bus_id)
        for bus_id in getattr(full_network, "household_bus_ids", [])
    }

    if not all_household_ids:
        return sorted(
            str(bus_id)
            for bus_id in getattr(network, "household_bus_ids", [])
        )

    ordered_household_ids: list[str] = []
    already_added: set[str] = set()

    for trafo_id in selectable_transformer_ids(full_network):
        try:
            reachable_bus_ids = reachable_bus_ids_for_transformer(
                full_network,
                trafo_id,
            )
        except Exception:
            continue

        households_in_area = [
            household_id
            for household_id in all_household_ids
            if household_id in reachable_bus_ids
            and household_id not in already_added
        ]

        households_in_area.sort(key=household_sort_key)

        for household_id in households_in_area:
            ordered_household_ids.append(household_id)
            already_added.add(household_id)

    remaining_households = [
        household_id
        for household_id in all_household_ids
        if household_id not in already_added
    ]

    remaining_households.sort(key=household_sort_key)
    ordered_household_ids.extend(remaining_households)

    return ordered_household_ids


def sort_household_ids_for_display(
    household_ids: list[str],
    reference_household_ids: list[str],
) -> list[str]:
    """
    Sort visible household IDs according to the global reference order.
    """
    reference_position = {
        str(household_id): index
        for index, household_id in enumerate(reference_household_ids)
    }

    return sorted(
        (str(household_id) for household_id in household_ids),
        key=lambda household_id: (
            reference_position.get(household_id, 10**12),
            household_sort_key(household_id),
        ),
    )


def household_display_label(household_ids: list[str], household_id: str) -> str:
    """
    Return a user-friendly household label while keeping the internal ID unchanged.
    """
    household_id = str(household_id)
    household_ids_in_order = [str(item) for item in household_ids]

    try:
        household_number = household_ids_in_order.index(household_id) + 1
    except ValueError:
        household_number = None

    building_id = extract_building_id(household_id)

    if household_number is None:
        if building_id:
            return f"Haushalt – Gebäude {building_id}"

        return "Haushalt"

    if building_id:
        return f"Haushalt {household_number} – Gebäude {building_id}"

    return f"Haushalt {household_number}"


def household_sort_key(household_id: str) -> tuple[int, int, str]:
    """
    Sort households by building number if available.
    """
    building_id = extract_building_id(household_id)

    if building_id is not None:
        try:
            return 0, int(building_id), str(household_id)
        except ValueError:
            pass

    return 1, 10**18, str(household_id)


def extract_building_id(bus_id: str) -> str | None:
    """
    Extract a building identifier from known household bus IDs.

    Preferred pattern:
    ..._building_1555885

    Fallback:
    use the last numeric part of the ID if no explicit building marker exists.
    """
    bus_id = str(bus_id)

    marker = "_building_"
    if marker in bus_id:
        building_id = bus_id.rsplit(marker, 1)[-1].strip()
        return building_id or None

    numeric_parts = re.findall(r"\d+", bus_id)

    if numeric_parts:
        return numeric_parts[-1]

    return None


def build_transformer_bus_label_map(network) -> dict[str, list[str]]:
    """
    Return transformer-bus labels keyed by bus ID.

    A Transformatorbereich is the selected network area.
    A Transformator-Bus is a bus/node belonging to a transformer.
    The wording is kept separate intentionally.
    """
    labels_by_bus: dict[str, list[str]] = {}

    lv_grid_to_selectable_id = selectable_transformer_id_by_lv_grid(network)

    for transformer in getattr(network, "transformers", []):
        trafo_id = str(transformer.trafo_id)
        info = parse_transformer_id(trafo_id)
        lv_grid_id = info["lv_grid_id"]

        representative_trafo_id = (
            lv_grid_to_selectable_id.get(lv_grid_id)
            if lv_grid_id is not None
            else trafo_id
        )

        if representative_trafo_id is None:
            representative_trafo_id = trafo_id

        area_label = transformer_area_display_label(network, representative_trafo_id)

        for bus_id in (transformer.hv_bus, transformer.lv_bus):
            key = str(bus_id)
            labels_by_bus.setdefault(key, [])

            if area_label not in labels_by_bus[key]:
                labels_by_bus[key].append(area_label)

    return labels_by_bus


def selectable_transformer_id_by_lv_grid(network) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for trafo_id in selectable_transformer_ids(network):
        info = parse_transformer_id(trafo_id)
        lv_grid_id = info["lv_grid_id"]

        if lv_grid_id is None:
            continue

        if lv_grid_id not in mapping:
            mapping[lv_grid_id] = trafo_id

    return mapping


def build_bus_label_map(
    network,
    household_label_map: dict[str, str],
    transformer_bus_labels: dict[str, list[str]],
) -> dict[str, str]:
    bus_label_map: dict[str, str] = {}
    other_bus_counter = 1

    for bus in getattr(network, "buses", []):
        bus_id = str(bus.bus_id)

        if bus_id in household_label_map:
            bus_label_map[bus_id] = household_label_map[bus_id]
            continue

        if bus_id in transformer_bus_labels:
            labels = transformer_bus_labels.get(bus_id, [])
            if labels:
                bus_label_map[bus_id] = f"Transformator-Bus – {labels[0]}"
            else:
                bus_label_map[bus_id] = "Transformator-Bus"
            continue

        bus_label_map[bus_id] = f"Netzknoten {other_bus_counter}"
        other_bus_counter += 1

    return bus_label_map