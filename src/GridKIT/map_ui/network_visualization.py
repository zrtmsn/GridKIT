# ─────────────────────────────────────────────────────────────
# map_ui/network_visualization.py
#
# Functions for visualizing a generated GridNetwork on a Folium map.
# This module contains the network map, line rendering, bus rendering,
# and popup content including household scenario values.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

import folium
import streamlit as st
from streamlit_folium import st_folium

from map_ui.household_config import (
    build_household_configuration,
    default_scenario_assumptions,
)
from map_ui.osm_fetcher import AreaBounds


def show_network_visualization(network, selected_bounds: AreaBounds) -> None:
    st.subheader("Netzansicht / Netzwerkkonfiguration")

    st.markdown(
        """
        **Legende**
        - 🟠 Transformator-Bus
        - 🟢 Haushalts-/Last-Bus
        - 🔵 Sonstiger Netzknoten
        - Türkise Linien = Leitungen
        """
    )

    household_configuration = build_current_household_configuration_from_session(
        network=network,
        selected_bounds=selected_bounds,
    )

    fmap, missing_bus_coords, missing_line_coords = create_network_map(
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

    info_col1, info_col2 = st.columns(2)
    info_col1.metric("Busse ohne Koordinaten", missing_bus_coords)
    info_col2.metric("Leitungen ohne vollständig darstellbare Endpunkte", missing_line_coords)

    with st.expander("Hinweis zur Visualisierung"):
        st.write(
            "Die Netzansicht wird direkt aus dem von `grid_model` erzeugten `GridNetwork` aufgebaut. "
            "Busse werden anhand ihrer Koordinaten dargestellt, Leitungen verbinden die zugehörigen Busse. "
            "Damit sie sichtbar sind, müssen für die Busse `x_coord` und `y_coord` vorhanden sein."
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
        "load_scaling_factor": float(load_scaling_by_bus[bus_id]),
        "is_individual_adjustment": bus_id in individual_adjustments,
    }


def format_bool_de(value: bool) -> str:
    return "Ja" if value else "Nein"


def create_network_map(
    network,
    selected_bounds: AreaBounds,
    household_configuration: dict[str, Any] | None = None,
) -> tuple[folium.Map, int, int]:
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

    transformer_bus_ids = set()
    for trafo in network.transformers:
        transformer_bus_ids.add(str(trafo.hv_bus))
        transformer_bus_ids.add(str(trafo.lv_bus))

    line_group = folium.FeatureGroup(name="Leitungen", show=True)
    household_group = folium.FeatureGroup(name="Haushalte/Lastpunkte", show=True)
    transformer_group = folium.FeatureGroup(name="Transformator-Busse", show=True)
    bus_group = folium.FeatureGroup(name="Sonstige Netzknoten", show=True)

    missing_line_coords = add_network_lines(line_group, network, bus_lookup)
    missing_bus_coords = add_network_buses(
        household_group=household_group,
        transformer_group=transformer_group,
        bus_group=bus_group,
        network=network,
        household_ids=household_ids,
        transformer_bus_ids=transformer_bus_ids,
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

    return fmap, missing_bus_coords, missing_line_coords


def add_network_lines(line_group, network, bus_lookup: dict[str, Any]) -> int:
    missing_line_coords = 0

    for line in network.lines:
        from_bus = bus_lookup.get(str(line.from_bus))
        to_bus = bus_lookup.get(str(line.to_bus))

        if from_bus is None or to_bus is None:
            missing_line_coords += 1
            continue

        if (
            from_bus.x_coord is None
            or from_bus.y_coord is None
            or to_bus.x_coord is None
            or to_bus.y_coord is None
        ):
            missing_line_coords += 1
            continue

        popup_html = f"""
        <b>Line ID:</b> {line.line_id}<br>
        <b>From:</b> {line.from_bus}<br>
        <b>To:</b> {line.to_bus}<br>
        <b>Length (km):</b> {line.length_km:.4f}<br>
        <b>R (Ohm/km):</b> {line.r_ohm_per_km:.4f}<br>
        <b>X (Ohm/km):</b> {line.x_ohm_per_km:.4f}<br>
        <b>Max I (kA):</b> {line.max_i_ka:.4f}
        """

        folium.PolyLine(
            locations=[
                [from_bus.y_coord, from_bus.x_coord],
                [to_bus.y_coord, to_bus.x_coord],
            ],
            color="#0f9d8a",
            weight=3,
            opacity=0.85,
            tooltip=f"Leitung: {line.line_id}",
            popup=folium.Popup(popup_html, max_width=350),
        ).add_to(line_group)

    return missing_line_coords


def add_network_buses(
    household_group,
    transformer_group,
    bus_group,
    network,
    household_ids: set[str],
    transformer_bus_ids: set[str],
    household_configuration: dict[str, Any] | None = None,
) -> int:
    missing_bus_coords = 0

    for bus in network.buses:
        if bus.x_coord is None or bus.y_coord is None:
            missing_bus_coords += 1
            continue

        bus_id = str(bus.bus_id)

        popup_html = make_bus_popup(
            bus=bus,
            household_ids=household_ids,
            transformer_bus_ids=transformer_bus_ids,
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
                tooltip=f"Transformator-Bus: {bus.bus_id}",
                popup=folium.Popup(popup_html, max_width=350),
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
                tooltip=f"Haushalt/Lastpunkt: {bus.bus_id}",
                popup=folium.Popup(popup_html, max_width=350),
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
                tooltip=f"Sonstiger Netzknoten: {bus.bus_id}",
                popup=folium.Popup(popup_html, max_width=350),
            ).add_to(bus_group)

    return missing_bus_coords


def make_bus_popup(
    bus,
    household_ids: set[str],
    transformer_bus_ids: set[str],
    network,
    household_configuration: dict[str, Any] | None = None,
) -> str:
    bus_id = str(bus.bus_id)
    roles: list[str] = []

    if bus_id in transformer_bus_ids:
        roles.append("Transformator-Bus")
    if bus_id in household_ids:
        roles.append("Haushalt/Lastpunkt")
    if not roles:
        roles.append("Sonstiger Netzknoten")

    has_ev_from_gridnetwork = bus_id in getattr(network, "ev_availability", {})
    has_load_profile = bus_id in getattr(network, "household_load_profile_kw", {})

    scenario_values = get_bus_scenario_values(
        bus_id=bus_id,
        household_configuration=household_configuration,
    )

    scenario_html = ""

    if scenario_values is not None:
        scenario_html = f"""
        <br>
        <b>EV im aktuellen Szenario:</b> {format_bool_de(scenario_values["has_ev"])}<br>
        <b>WP im aktuellen Szenario:</b> {format_bool_de(scenario_values["has_heat_pump"])}<br>
        <b>Verbrauchsfaktor im Szenario:</b> {scenario_values["load_scaling_factor"]:.2f}<br>
        <b>Individuell angepasst:</b> {format_bool_de(scenario_values["is_individual_adjustment"])}
        """

    return f"""
    <b>Bus ID:</b> {bus.bus_id}<br>
    <b>Rolle:</b> {", ".join(roles)}<br>
    <b>Nominalspannung (kV):</b> {bus.v_nom_kv}<br>
    <b>x_coord:</b> {bus.x_coord}<br>
    <b>y_coord:</b> {bus.y_coord}<br>
    <b>Load Profile:</b> {"Ja" if has_load_profile else "Nein"}<br>
    <b>EV Availability aus GridNetwork:</b> {"Ja" if has_ev_from_gridnetwork else "Nein"}
    {scenario_html}
    """