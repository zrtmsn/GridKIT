# ─────────────────────────────────────────────────────────────
# map_ui/map_widget.py
#
# Streamlit + Folium GUI for selecting an area and displaying
# GridNetwork information from grid_model's OSMNetworkBuilder.
#
# Household configuration and training UI sections live in
# map_ui/map_widget_sections.py to keep this file readable.
#
# Run from project root:
#   python -m streamlit run src/GridKIT/map_ui/map_widget.py
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

import folium
import requests
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from grid_model.builder import OSMNetworkBuilder
from map_ui.household_config import default_scenario_assumptions
from map_ui.map_widget_sections import (
    show_household_configuration,
    show_training_section,
)
from map_ui.network_visualization import show_network_visualization
from map_ui.osm_fetcher import AreaBounds
from map_ui.transformer_network_filter import (
    device_summary,
    show_transformer_selection,
)


DEFAULT_CENTER = (49.0069, 8.4037)  # Karlsruhe
DEFAULT_ZOOM = 15

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "GridKIT-map-ui/0.1"


def render_map_ui() -> None:
    """Page body without st.set_page_config.

    This function can be used as one page of the integrated app via
    scripts/app.py and can also run standalone through main().
    """
    st.title("GridKIT Karte")
    st.caption(
        "Ort suchen → Bereich auswählen → GridNetwork erzeugen → "
        "optional Transformator auswählen → Netzansicht visualisieren → "
        "Haushalte konfigurieren → speichern & trainieren"
    )

    initialise_session_state()

    with st.sidebar:
        st.header("Eingabe")
        st.write(
            "Suche zuerst einen Ort oder zeichne direkt links ein Rechteck/Polygon. "
            "Alternativ kannst du unten eine Bounding Box manuell eingeben."
        )

        st.subheader("Ort suchen")

        place_query = st.text_input(
            "Ort, Stadt, Straße oder Adresse",
            placeholder="z. B. Karlsruhe, Kaiserstraße Karlsruhe",
        )

        search_clicked = st.button("Ort suchen")

        if search_clicked:
            try:
                st.session_state["search_results"] = search_place(place_query)
                if not st.session_state["search_results"]:
                    st.warning("Keine Suchergebnisse gefunden.")
            except Exception as exc:
                st.error("Die Ortssuche ist fehlgeschlagen.")
                st.exception(exc)
                st.session_state["search_results"] = []

        search_results = st.session_state.get("search_results", [])

        if search_results:
            selected_index = st.selectbox(
                "Suchergebnis auswählen",
                options=list(range(len(search_results))),
                format_func=lambda i: format_search_result(search_results[i]),
            )

            selected_place = search_results[selected_index]

            if st.button("Karte auf Suchergebnis zentrieren"):
                lat = float(selected_place["lat"])
                lon = float(selected_place["lon"])

                st.session_state["map_center"] = (lat, lon)
                st.session_state["map_zoom"] = 16
                st.session_state["search_marker"] = {
                    "lat": lat,
                    "lon": lon,
                    "display_name": selected_place.get("display_name", "Suchergebnis"),
                }

                st.rerun()

        st.divider()

        area_name = st.text_input("area_name", value="selected_area")

        st.subheader("Manuelle Bounding Box")
        south = st.number_input("south / min latitude", value=49.0000, format="%.6f")
        west = st.number_input("west / min longitude", value=8.3900, format="%.6f")
        north = st.number_input("north / max latitude", value=49.0100, format="%.6f")
        east = st.number_input("east / max longitude", value=8.4100, format="%.6f")

        use_manual_bbox = st.checkbox("Manuelle Bounding Box verwenden", value=False)

        st.subheader("grid_model")
        max_households = st.number_input(
            "Maximale Anzahl Haushalte pro Feeder",
            value=15,
            min_value=1,
            max_value=200,
            step=1,
        )

        conda_env = st.text_input("GridCreator Conda Environment", value="GridCreator")

        build_clicked = st.button("GridNetwork erzeugen", type="primary")

    col_map, col_out = st.columns([3, 2])

    with col_map:
        st.subheader("Bereichsauswahl")

        fmap = make_base_map(
            center=st.session_state["map_center"],
            zoom=st.session_state["map_zoom"],
            search_marker=st.session_state.get("search_marker"),
        )

        map_key = (
            f"base_map_"
            f"{st.session_state['map_center'][0]:.6f}_"
            f"{st.session_state['map_center'][1]:.6f}_"
            f"{st.session_state['map_zoom']}"
        )

        map_data = st_folium(
            fmap,
            height=650,
            width=None,
            returned_objects=["last_active_drawing", "all_drawings"],
            key=map_key,
        )

    selected_bounds: AreaBounds | None = None

    if use_manual_bbox:
        try:
            selected_bounds = AreaBounds(
                south=south,
                west=west,
                north=north,
                east=east,
            )
        except Exception as exc:
            st.error(f"Ungültige Bounding Box: {exc}")
            selected_bounds = None
    else:
        selected_bounds = bounds_from_drawings(map_data)

    with col_out:
        st.subheader("Auswahl")

        if selected_bounds:
            st.json(selected_bounds.model_dump())
            st.metric("Fläche ca. km²", f"{selected_bounds.approx_area_km2():.3f}")
        else:
            st.info("Noch kein Bereich ausgewählt. Zeichne ein Rechteck/Polygon auf der Karte.")

        if build_clicked:
            if not selected_bounds:
                st.error("Bitte zuerst einen Bereich auswählen.")
                return

            scenario = area_name.strip() or "selected_area"

            with st.spinner("GridNetwork wird erzeugt ..."):
                try:
                    builder = OSMNetworkBuilder(
                        top=selected_bounds.north,
                        bottom=selected_bounds.south,
                        left=selected_bounds.west,
                        right=selected_bounds.east,
                        scenario=scenario,
                        conda_env=conda_env.strip() or "GridCreator",
                    )

                    full_network = builder.build()

                    st.session_state["full_network"] = full_network
                    st.session_state["built_network"] = full_network
                    st.session_state["built_bounds"] = selected_bounds
                    st.session_state["built_scenario"] = scenario
                    st.session_state["max_households"] = int(max_households)

                    # Start with the complete generated GridNetwork. A transformer
                    # filter is only applied after the user explicitly selects one.
                    st.session_state["selected_trafo_id"] = None

                    # Reset all interactive configuration state for a new network.
                    st.session_state["household_overrides"] = {}
                    st.session_state["global_device_targets"] = None
                    st.session_state["global_device_target_scope"] = None
                    st.session_state["household_config_version"] += 1
                    st.session_state["scenario_config_version"] += 1

                except Exception as exc:
                    if is_overpass_timeout_error(exc):
                        st.error(
                            "Die Overpass API ist aktuell nicht erreichbar oder antwortet zu langsam.\n\n"
                            "Bitte versuchen Sie es später erneut oder wählen Sie ein kleineres Gebiet."
                        )

                        with st.expander("Technische Details anzeigen"):
                            st.exception(exc)
                    else:
                        st.error(
                            "Das GridNetwork konnte nicht erzeugt werden. "
                            "Bitte prüfen Sie die Eingaben oder versuchen Sie es erneut."
                        )

                        with st.expander("Technische Details anzeigen"):
                            st.exception(exc)

                    return

            st.success("GridNetwork erzeugt.")

    full_network = st.session_state.get("full_network")
    built_bounds = st.session_state.get("built_bounds")

    if full_network is not None and built_bounds is not None:
        st.divider()
        built_network = show_transformer_selection(full_network)
        st.divider()
        show_grid_model_result(built_network, built_bounds)
        st.divider()
        show_network_visualization(built_network, built_bounds)
        st.divider()
        household_configuration = show_household_configuration(built_network, built_bounds)

        if household_configuration is not None:
            st.divider()
            show_training_section(built_network, household_configuration)


def initialise_session_state() -> None:
    if "full_network" not in st.session_state:
        st.session_state["full_network"] = None

    if "built_network" not in st.session_state:
        st.session_state["built_network"] = None

    if "built_bounds" not in st.session_state:
        st.session_state["built_bounds"] = None

    if "built_scenario" not in st.session_state:
        st.session_state["built_scenario"] = None

    if "selected_trafo_id" not in st.session_state:
        st.session_state["selected_trafo_id"] = None

    if "map_center" not in st.session_state:
        st.session_state["map_center"] = DEFAULT_CENTER

    if "map_zoom" not in st.session_state:
        st.session_state["map_zoom"] = DEFAULT_ZOOM

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []

    if "search_marker" not in st.session_state:
        st.session_state["search_marker"] = None

    if "household_overrides" not in st.session_state:
        st.session_state["household_overrides"] = {}

    if "scenario_assumptions" not in st.session_state:
        st.session_state["scenario_assumptions"] = default_scenario_assumptions()

    if "global_device_targets" not in st.session_state:
        st.session_state["global_device_targets"] = None

    if "global_device_target_scope" not in st.session_state:
        st.session_state["global_device_target_scope"] = None

    if "scenario_config_version" not in st.session_state:
        st.session_state["scenario_config_version"] = 0

    if "household_config_version" not in st.session_state:
        st.session_state["household_config_version"] = 0


def make_base_map(
    center: tuple[float, float] = DEFAULT_CENTER,
    zoom: int = DEFAULT_ZOOM,
    search_marker: dict[str, Any] | None = None,
) -> folium.Map:
    fmap = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    if search_marker:
        folium.Marker(
            location=[search_marker["lat"], search_marker["lon"]],
            tooltip="Suchergebnis",
            popup=folium.Popup(search_marker["display_name"], max_width=350),
        ).add_to(fmap)

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "rectangle": True,
            "polygon": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)

    return fmap


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def search_place(query: str) -> list[dict[str, Any]]:
    query = query.strip()

    if not query:
        return []

    response = requests.get(
        NOMINATIM_SEARCH_URL,
        params={
            "q": query,
            "format": "jsonv2",
            "limit": 5,
            "addressdetails": 1,
            "countrycodes": "de",
        },
        headers={
            "User-Agent": NOMINATIM_USER_AGENT,
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def format_search_result(result: dict[str, Any]) -> str:
    return result.get("display_name", "Unbekanntes Suchergebnis")


def is_overpass_timeout_error(exc: Exception) -> bool:
    error_text = str(exc).lower()

    overpass_indicators = [
        "overpass-api",
        "overpass",
    ]

    timeout_indicators = [
        "connecttimeout",
        "readtimeout",
        "timed out",
        "timeout",
        "max retries exceeded",
        "connection aborted",
        "connection error",
        "connection reset",
    ]

    has_overpass_reference = any(
        indicator in error_text for indicator in overpass_indicators
    )

    has_timeout_reference = any(
        indicator in error_text for indicator in timeout_indicators
    )

    return has_overpass_reference and has_timeout_reference


def bounds_from_drawings(map_data: dict[str, Any] | None) -> AreaBounds | None:
    if not map_data:
        return None

    drawing = map_data.get("last_active_drawing")

    if not drawing:
        drawings = map_data.get("all_drawings") or []
        drawing = drawings[-1] if drawings else None

    if not drawing:
        return None

    geometry = drawing.get("geometry") or {}
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    if not coordinates:
        return None

    lon_lat_pairs: list[tuple[float, float]] = []

    if geom_type == "Polygon":
        lon_lat_pairs = [(float(lon), float(lat)) for lon, lat in coordinates[0]]

    elif geom_type == "MultiPolygon":
        for polygon in coordinates:
            lon_lat_pairs.extend((float(lon), float(lat)) for lon, lat in polygon[0])

    if not lon_lat_pairs:
        return None

    lons = [p[0] for p in lon_lat_pairs]
    lats = [p[1] for p in lon_lat_pairs]

    return AreaBounds(
        south=min(lats),
        west=min(lons),
        north=max(lats),
        east=max(lons),
    )


def show_grid_model_result(network, selected_bounds: AreaBounds) -> None:
    st.subheader("GridNetwork")

    selected_trafo_id = st.session_state.get("selected_trafo_id")

    if selected_trafo_id is None:
        st.caption(
            "Es ist aktuell kein Trafo-Filter aktiv. "
            "Die Netzwerkkonfiguration zeigt das vollständig erzeugte GridNetwork "
            "für den ausgewählten Kartenbereich."
        )
    else:
        st.caption(
            f"Aktiver Trafo-Filter: Die Netzwerkkonfiguration ist auf Transformator "
            f"{selected_trafo_id} begrenzt. Der JSON-Export enthält nur dieses "
            f"topologisch gefilterte Trafo-Teilnetz."
        )

    bus_count = len(network.buses)
    line_count = len(network.lines)
    transformer_count = len(network.transformers)
    household_count = len(network.household_bus_ids)
    devices = device_summary(network)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buses / Knoten", bus_count)
    c2.metric("Lines / Kanten", line_count)
    c3.metric("Households", household_count)
    c4.metric("Transformers", transformer_count)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("EV aus GridCreator", devices["ev_count"])
    d2.metric("WP aus GridCreator", devices["heat_pump_count"])
    d3.metric("Batterien aus GridCreator", devices["battery_count"])
    d4.metric("PV aus GridCreator", devices["pv_count"])

    st.write("**Ausgewählter Kartenbereich**")
    st.json(
        {
            "bbox": selected_bounds.model_dump(),
            "area_km2": selected_bounds.approx_area_km2(),
        }
    )

    filtering_mode = (
        "complete_network"
        if selected_trafo_id is None
        else "topology_based_transformer_feeder"
    )

    st.write("**Von grid_model erzeugte Netzwerkdaten**")
    st.json(
        {
            "network_id": network.network_id,
            "area_name": getattr(network, "area_name", None),
            "bus_count": bus_count,
            "line_count": line_count,
            "household_count": household_count,
            "transformer_count": transformer_count,
            "selected_trafo_id": selected_trafo_id,
            "filtering": filtering_mode,
            "gridcreator_device_summary": devices,
        }
    )

    grid_json = network.model_dump_json(indent=2)

    st.download_button(
        label="grid_network.json herunterladen",
        data=grid_json,
        file_name="grid_network.json",
        mime="application/json",
    )

    with st.expander("GridNetwork JSON anzeigen"):
        st.code(grid_json, language="json")


def main() -> None:
    """Standalone entry point: streamlit run src/GridKIT/map_ui/map_widget.py."""
    st.set_page_config(page_title="GridKIT map_ui", layout="wide")
    render_map_ui()


if __name__ == "__main__":
    main()