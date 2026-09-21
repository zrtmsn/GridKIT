from __future__ import annotations

from typing import Any

import folium
import streamlit as st
from streamlit_folium import st_folium

from map_ui.transformer_network_filter_helpers import (
    ALL_NETWORK_OPTION,
    assigned_household_ids_for_transformer,
    bus_by_id,
    bus_ids_for_lv_grid,
    device_summary,
    device_summary_for_households,
    distance_m,
    filter_network_by_transformer,
    format_transformer_option,
    household_ids_for_lv_grid,
    line_adjacency,
    lv_grid_id_from_bus_id,
    map_center_from_transformers_or_network,
    nearest_transformer_id,
    parse_transformer_id,
    percent,
    reachable_bus_ids_for_transformer,
    read_object_value,
    selectable_transformer_ids,
    selectable_transformers,
    transformer_area_display_label,
    transformer_by_id,
    transformer_display_coordinates,
    transformer_display_label,
    transformer_group_members,
    transformer_marker_rows,
    transformer_metadata_for_id,
)


def show_transformer_selection(full_network):
    """
    Render the transformer selection UI for a generated GridNetwork.

    Initial state:
    - no transformer is selected
    - the complete generated GridNetwork is displayed

    After a transformer is selected:
    - the network is filtered to the selected transformer area
    - households are assigned using model/source identifiers
    - topology traversal is still used for connected non-household buses and lines
    - only assigned households, matching household data and related entries remain

    Reinforcement transformers are kept internally in the GridNetwork, but they are
    not shown as separate user-facing selection options.
    """
    st.subheader("Trafo-Auswahl")

    if not getattr(full_network, "transformers", None):
        st.warning("Im erzeugten GridNetwork wurde kein Transformator gefunden.")
        st.session_state["selected_trafo_id"] = None
        st.session_state["built_network"] = full_network
        return full_network

    trafo_options = selectable_transformer_ids(full_network)

    if not trafo_options:
        st.warning(
            "Im erzeugten GridNetwork wurden Transformatoren gefunden, aber keine "
            "regulären Transformatorbereiche für die Auswahl erkannt."
        )
        st.session_state["selected_trafo_id"] = None
        st.session_state["built_network"] = full_network
        return full_network

    current_selected = st.session_state.get("selected_trafo_id")
    if current_selected not in trafo_options:
        current_selected = None
        st.session_state["selected_trafo_id"] = None

    marker_rows = transformer_marker_rows(full_network)

    if marker_rows:
        st.caption(
            "Am Anfang wird das gesamte erzeugte Netz angezeigt. "
            "Wählen Sie einen Transformatorbereich auf der Karte oder über die Auswahlbox aus, "
            "um nur diesen Netzbereich anzuzeigen."
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
            "Transformatorbereiche wurden im GridNetwork gefunden, aber ohne Koordinaten. "
            "Die Auswahl ist deshalb nur über die Auswahlbox möglich."
        )

    options = [ALL_NETWORK_OPTION] + trafo_options

    selected_option = current_selected if current_selected is not None else ALL_NETWORK_OPTION
    selected_index = options.index(selected_option)

    selected_option = st.selectbox(
        "Netzkonfiguration auswählen",
        options=options,
        index=selected_index,
        format_func=lambda option: format_transformer_option(option, full_network),
        help=(
            "Mit „Gesamtes Netz anzeigen“ wird die vollständige erzeugte Netzwerkkonfiguration angezeigt. "
            "Bei Auswahl eines Transformatorbereichs wird das Netz auf diesen Bereich gefiltert. "
            "Die Haushaltszuordnung erfolgt anhand der eindeutigen Zuordnung aus den Modelldaten. "
            "Technische Zusatztransformatoren werden intern berücksichtigt, aber nicht als eigene "
            "Auswahloption angezeigt."
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
            f"Die folgenden Abschnitte zeigen das vollständig erzeugte Netz für den "
            f"ausgewählten Kartenbereich. Dazu gehören alle enthaltenen Knoten, Leitungen, "
            f"Haushalte und Transformatorbereiche.\n\n"
            f"Umfang des vollständigen Netzes: "
            f"**{len(full_network.buses)} Knoten**, "
            f"**{len(full_network.lines)} Leitungen**, "
            f"**{len(full_network.household_bus_ids)} Haushalte**, "
            f"**{len(trafo_options)} auswählbare Transformatorbereiche**.\n\n"
            f"Erkannte Ausstattung: "
            f"**{full_device_summary['ev_count']} Haushalte mit Elektroauto**, "
            f"**{full_device_summary['heat_pump_count']} Haushalte mit Wärmepumpe**, "
            f"**{full_device_summary['battery_count']} Haushalte mit Batteriespeicher**, "
            f"**{full_device_summary['pv_count']} Haushalte mit PV-Anlage**."
        )

        with st.expander("Verfügbare Transformatorbereiche anzeigen"):
            st.json(
                {
                    "mode": "complete_network",
                    "selected_trafo_id": None,
                    "available_transformer_areas": [
                        transformer_metadata_for_id(trafo_id)
                        for trafo_id in trafo_options
                    ],
                    "internal_transformer_count": len(full_network.transformers),
                    "selectable_transformer_area_count": len(trafo_options),
                    "filtering": "none",
                    "device_summary": full_device_summary,
                }
            )

        return full_network

    filtered_network = filter_network_by_transformer(full_network, active_trafo_id)
    st.session_state["built_network"] = filtered_network

    selected_transformer = transformer_by_id(full_network, active_trafo_id)
    selected_transformer_label = transformer_area_display_label(
        full_network,
        selected_transformer.trafo_id,
    )
    selected_transformer_info = parse_transformer_id(selected_transformer.trafo_id)
    filtered_device_summary = device_summary(filtered_network)

    st.info(
        f"**Aktiver Trafo-Filter:** **{selected_transformer_label}** wurde ausgewählt.\n\n"
        f"Die folgenden Abschnitte beziehen sich nun auf diesen Transformatorbereich. "
        f"Die Netzwerkkonfiguration, die Netzvisualisierung, die Haushaltskonfiguration "
        f"sowie der JSON-Export enthalten nur die Knoten, Leitungen und Haushalte, "
        f"die diesem Bereich zugeordnet wurden.\n\n"
        f"Umfang des ausgewählten Transformatorbereichs: "
        f"**{len(filtered_network.buses)} Knoten**, "
        f"**{len(filtered_network.lines)} Leitungen**, "
        f"**{len(filtered_network.household_bus_ids)} Haushalte**.\n\n"
        f"Erkannte Ausstattung in diesem Bereich: "
        f"**{filtered_device_summary['ev_count']} Haushalte mit Elektroauto**, "
        f"**{filtered_device_summary['heat_pump_count']} Haushalte mit Wärmepumpe**, "
        f"**{filtered_device_summary['battery_count']} Haushalte mit Batteriespeicher**, "
        f"**{filtered_device_summary['pv_count']} Haushalte mit PV-Anlage**."
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
                "included_transformers": [
                    transformer_metadata_for_id(transformer.trafo_id)
                    for transformer in transformer_group_members(full_network, active_trafo_id)
                ],
                "hv_bus": selected_transformer.hv_bus,
                "lv_bus": selected_transformer.lv_bus,
                "s_nom_mva": selected_transformer.s_nom_mva,
                "vn_hv_kv": selected_transformer.vn_hv_kv,
                "vn_lv_kv": selected_transformer.vn_lv_kv,
                "filtering": "topology_and_model_id_based",
                "household_assignment": "model_source_identifier_based",
                "device_summary": filtered_device_summary,
            }
        )

    return filtered_network


def apply_network_selection(full_network, trafo_id: str | None) -> None:
    """
    Apply either the complete network or a transformer-filtered network.
    """
    previous_trafo_id = st.session_state.get("selected_trafo_id")

    if trafo_id is None:
        selected_network = full_network
    else:
        selected_network = filter_network_by_transformer(full_network, trafo_id)

    st.session_state["selected_trafo_id"] = trafo_id
    st.session_state["built_network"] = selected_network

    if previous_trafo_id != trafo_id:
        st.session_state["household_overrides"] = {}
        st.session_state["global_device_targets"] = None
        st.session_state["global_device_target_scope"] = None

        st.session_state.setdefault("household_config_version", 0)
        st.session_state["household_config_version"] += 1

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
            else row["display_label"]
        )

        lv_grid_label = row["lv_grid_id"] if row["lv_grid_id"] is not None else "Unbekannt"

        popup_html = f"""
        <b>{row["display_label"]}</b><br>
        LV-Grid: {lv_grid_label}<br>
        Haushalte im Transformatorbereich: {row["household_count"]}<br>
        Knoten im Transformatorbereich: {row["bus_count"]}<br>
        Haushalte mit Elektroauto: {row["ev_count"]}<br>
        Haushalte mit Wärmepumpe: {row["heat_pump_count"]}<br>
        Haushalte mit Batteriespeicher: {row["battery_count"]}<br>
        Haushalte mit PV-Anlage: {row["pv_count"]}<br>
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


__all__ = [
    "ALL_NETWORK_OPTION",
    "show_transformer_selection",
    "apply_network_selection",
    "make_transformer_selection_map",
    "assigned_household_ids_for_transformer",
    "bus_by_id",
    "bus_ids_for_lv_grid",
    "device_summary",
    "device_summary_for_households",
    "distance_m",
    "filter_network_by_transformer",
    "format_transformer_option",
    "household_ids_for_lv_grid",
    "line_adjacency",
    "lv_grid_id_from_bus_id",
    "map_center_from_transformers_or_network",
    "nearest_transformer_id",
    "parse_transformer_id",
    "percent",
    "reachable_bus_ids_for_transformer",
    "read_object_value",
    "selectable_transformer_ids",
    "selectable_transformers",
    "transformer_area_display_label",
    "transformer_by_id",
    "transformer_display_coordinates",
    "transformer_display_label",
    "transformer_group_members",
    "transformer_marker_rows",
    "transformer_metadata_for_id",
]
