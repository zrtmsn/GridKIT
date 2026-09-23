from __future__ import annotations

import folium
import streamlit as st
from streamlit_folium import st_folium

from map_ui.transformer_network_filter_helpers import (
    ALL_NETWORK_OPTION,
    assigned_household_ids_for_transformer,
    bus_by_id,
    device_summary,
    device_summary_for_households,
    distance_m,
    filter_network_by_transformer,
    format_transformer_option,
    line_adjacency,
    map_center_from_transformers_or_network,
    nearest_transformer_id,
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


TRANSFORMER_SELECTION_MAP_HEIGHT_PX = 420
TRANSFORMER_SELECTION_MAP_ZOOM = 15
TRANSFORMER_MARKER_CLICK_DISTANCE_M = 50.0
TRANSFORMER_POPUP_MAX_WIDTH = 400

INITIAL_CONFIG_VERSION = 0
REINFORCED_TRANSFORMER_COUNT_THRESHOLD = 1

COMPLETE_NETWORK_MAP_KEY_SUFFIX = "complete_network"
MAP_TILES = "OpenStreetMap"

MARKER_COLOR_SELECTED = "orange"
MARKER_COLOR_DEFAULT = "blue"
MARKER_ICON_NAME = "bolt"
MARKER_ICON_PREFIX = "fa"

MODE_COMPLETE_NETWORK = "complete_network"
MODE_TRANSFORMER_FEEDER = "transformer_feeder"

FILTERING_NONE = "none"
FILTERING_TOPOLOGY_BASED = "topology_based"
HOUSEHOLD_ASSIGNMENT_TOPOLOGY_BASED = "topology_based"


def show_transformer_selection(full_network):
    """
    Render the transformer selection UI for a generated GridNetwork.

    The complete network is shown by default. When a transformer area is
    selected, the displayed network is filtered by topology.
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

        map_key_suffix = (
            current_selected
            if current_selected is not None
            else COMPLETE_NETWORK_MAP_KEY_SUFFIX
        )

        clicked_data = st_folium(
            transformer_map,
            height=TRANSFORMER_SELECTION_MAP_HEIGHT_PX,
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
                    max_distance_m=TRANSFORMER_MARKER_CLICK_DISTANCE_M,
                )

                if (
                    clicked_trafo_id
                    and clicked_trafo_id != st.session_state.get("selected_trafo_id")
                ):
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
        format_func=format_transformer_option,
        help=(
            "Mit „Gesamtes Netz anzeigen“ wird die vollständige erzeugte Netzwerkkonfiguration angezeigt. "
            "Bei Auswahl eines Transformatorbereichs wird das Netz auf diesen Bereich gefiltert. "
            "Die Haushalte werden anhand der erzeugten Netzstruktur aus dem Modell zugeordnet. "
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
            f"**{len(full_network.transformers)} interne Transformatoren**, "
            f"**{len(trafo_options)} auswählbare Transformatorbereiche**.\n\n"
            f"Hinweis: Die Anzahl der internen Transformatoren kann höher sein als die Anzahl "
            f"der auswählbaren Transformatorbereiche, weil parallele oder verstärkte Transformatoren "
            f"mit demselben Niederspannungsbus gemeinsam als ein Transformatorbereich angezeigt werden.\n\n"
            f"Erkannte Ausstattung: "
            f"**{full_device_summary['ev_count']} Haushalte mit Elektroauto**, "
            f"**{full_device_summary['heat_pump_count']} Haushalte mit Wärmepumpe**, "
            f"**{full_device_summary['battery_count']} Haushalte mit Batteriespeicher**, "
            f"**{full_device_summary['pv_count']} Haushalte mit PV-Anlage**."
        )

        with st.expander("Verfügbare Transformatorbereiche anzeigen"):
            st.json(
                {
                    "mode": MODE_COMPLETE_NETWORK,
                    "selected_trafo_id": None,
                    "available_transformer_areas": [
                        transformer_metadata_for_id(trafo_id, full_network)
                        for trafo_id in trafo_options
                    ],
                    "internal_transformer_count": len(full_network.transformers),
                    "selectable_transformer_area_count": len(trafo_options),
                    "filtering": FILTERING_NONE,
                    "device_summary": full_device_summary,
                }
            )

        return full_network

    filtered_network = filter_network_by_transformer(full_network, active_trafo_id)
    st.session_state["built_network"] = filtered_network

    selected_transformer = transformer_by_id(full_network, active_trafo_id)
    selected_transformer_label = transformer_area_display_label(selected_transformer.trafo_id)
    filtered_device_summary = device_summary(filtered_network)

    st.info(
        f"**Aktiver Trafo-Filter:** **{selected_transformer_label}** wurde ausgewählt.\n\n"
        f"Die folgenden Abschnitte beziehen sich nun auf diesen Transformatorbereich. "
        f"Die Netzwerkkonfiguration, die Netzvisualisierung, die Haushaltskonfiguration "
        f"sowie der JSON-Export enthalten nur die Knoten, Leitungen und Haushalte, "
        f"die diesem Bereich über die erzeugte Netzstruktur zugeordnet wurden.\n\n"
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
                "mode": MODE_TRANSFORMER_FEEDER,
                "trafo_id": selected_transformer.trafo_id,
                "display_label": selected_transformer_label,
                "is_reinforced": (
                    len(transformer_group_members(full_network, active_trafo_id))
                    > REINFORCED_TRANSFORMER_COUNT_THRESHOLD
                ),
                "included_transformers": [
                    transformer_metadata_for_id(transformer.trafo_id, full_network)
                    for transformer in transformer_group_members(full_network, active_trafo_id)
                ],
                "hv_bus": selected_transformer.hv_bus,
                "lv_bus": selected_transformer.lv_bus,
                "s_nom_mva": selected_transformer.s_nom_mva,
                "vn_hv_kv": selected_transformer.vn_hv_kv,
                "vn_lv_kv": selected_transformer.vn_lv_kv,
                "filtering": FILTERING_TOPOLOGY_BASED,
                "household_assignment": HOUSEHOLD_ASSIGNMENT_TOPOLOGY_BASED,
                "device_summary": filtered_device_summary,
            }
        )

    return filtered_network


def apply_network_selection(full_network, trafo_id: str | None) -> None:
    """
    Apply either the complete network or a transformer-filtered network.

    Changing the selected transformer resets scenario and household overrides,
    because the visible household set may change.
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

        st.session_state.setdefault("household_config_version", INITIAL_CONFIG_VERSION)
        st.session_state["household_config_version"] += 1

        st.session_state.setdefault("scenario_config_version", INITIAL_CONFIG_VERSION)
        st.session_state["scenario_config_version"] += 1


def make_transformer_selection_map(
    full_network,
    selected_trafo_id: str | None = None,
) -> folium.Map:
    """
    Build a Folium map with clickable transformer markers.

    Marker clicks are only used for selecting a transformer area in the UI.
    Household assignment remains topology-based.
    """
    rows = transformer_marker_rows(full_network)
    center = map_center_from_transformers_or_network(full_network)

    fmap = folium.Map(
        location=center,
        zoom_start=TRANSFORMER_SELECTION_MAP_ZOOM,
        tiles=MAP_TILES,
        control_scale=True,
    )

    for row in rows:
        is_selected = selected_trafo_id is not None and row["trafo_id"] == selected_trafo_id

        icon_color = MARKER_COLOR_SELECTED if is_selected else MARKER_COLOR_DEFAULT
        tooltip = (
            f"Ausgewählt: {row['display_label']}"
            if is_selected
            else row["display_label"]
        )

        popup_html = f"""
        <b>{row["display_label"]}</b><br>
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
            popup=folium.Popup(popup_html, max_width=TRANSFORMER_POPUP_MAX_WIDTH),
            icon=folium.Icon(
                color=icon_color,
                icon=MARKER_ICON_NAME,
                prefix=MARKER_ICON_PREFIX,
            ),
        ).add_to(fmap)

    return fmap


__all__ = [
    "ALL_NETWORK_OPTION",
    "show_transformer_selection",
    "apply_network_selection",
    "make_transformer_selection_map",
    "assigned_household_ids_for_transformer",
    "bus_by_id",
    "device_summary",
    "device_summary_for_households",
    "distance_m",
    "filter_network_by_transformer",
    "format_transformer_option",
    "line_adjacency",
    "map_center_from_transformers_or_network",
    "nearest_transformer_id",
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
