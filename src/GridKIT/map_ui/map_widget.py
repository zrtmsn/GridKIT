# ─────────────────────────────────────────────────────────────
# map_ui/map_widget.py
#
# Streamlit + Folium GUI for selecting an area and displaying
# GridNetwork information from grid_model's OSMNetworkBuilder.
#
# Household configuration and training UI sections are imported
# directly from their dedicated files.
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
from map_ui.area_bounds import AreaBounds
from map_ui.household_config import default_scenario_assumptions
from map_ui.map_widget_households import show_household_configuration
from map_ui.map_widget_training import show_training_section
from map_ui.network_visualization import show_network_visualization
from map_ui.transformer_network_filter import (
    device_summary,
    show_transformer_selection,
)


DEFAULT_CENTER = (49.0069, 8.4037)  # Karlsruhe
DEFAULT_ZOOM = 15
SEARCH_RESULT_ZOOM = 16

DEFAULT_AREA_NAME = "ausgewaehlter_bereich"
FALLBACK_SCENARIO_NAME = "selected_area"

DEFAULT_MANUAL_SOUTH = 49.0000
DEFAULT_MANUAL_WEST = 8.3900
DEFAULT_MANUAL_NORTH = 49.0100
DEFAULT_MANUAL_EAST = 8.4100

MAP_COLUMN_RATIO = [3, 2]
BASE_MAP_HEIGHT_PX = 650
AREA_DISPLAY_DECIMALS = 3

GRID_METRIC_COLUMN_COUNT = 4
DEVICE_METRIC_COLUMN_COUNT = 4

SEARCH_MARKER_POPUP_MAX_WIDTH = 350

MANUAL_BBOX_COLOR = "red"
MANUAL_BBOX_WEIGHT = 2
MANUAL_BBOX_FILL = False
MANUAL_BBOX_TOOLTIP = "Manuelle Bounding Box"

SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "GridKIT-map-ui/0.1"
NOMINATIM_RESULT_LIMIT = 5
NOMINATIM_ADDRESS_DETAILS = 1
NOMINATIM_REQUEST_TIMEOUT_SECONDS = 10
NOMINATIM_COUNTRY_CODES = "de"
NOMINATIM_CACHE_TTL_SECONDS = SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY

DRAWING_FIRST_RING_INDEX = 0
DRAWING_LAST_ITEM_INDEX = -1

INITIAL_CONFIG_VERSION = 0

FILTERING_MODE_COMPLETE_NETWORK = "complete_network"
FILTERING_MODE_TRANSFORMER_AREA = "topology_based_transformer_area"

# Internal grid_model defaults.
# These values are intentionally not shown in the Map UI because they are
# technical developer settings and should not be changed from the map view.
INTERNAL_MAX_HOUSEHOLDS_PER_FEEDER = 15
GRIDCREATOR_CONDA_ENV = "GridCreator"


def render_map_ui() -> None:
    """
    Render the main map UI page.

    This function can be used as one page of the integrated app via
    scripts/app.py and can also run standalone through main().
    """
    st.title("GridKIT Karte")
    st.caption(
        "Ort suchen → Bereich auswählen → GridNetwork erzeugen → "
        "optional Transformator auswählen → Netzansicht visualisieren → "
        "Haushalte konfigurieren → Speichern & Training starten"
    )

    initialise_session_state()

    with st.sidebar:
        st.header("Eingabe")
        st.write(
            "Suchen Sie zunächst einen Ort, um die Karte auf den gewünschten Bereich zu zentrieren. "
            "Markieren Sie anschließend den Netzbereich durch Zeichnen eines Rechtecks oder Polygons "
            "auf der Karte. Alternativ können Sie die Koordinaten des Bereichs manuell als Bounding Box eingeben."
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
                st.session_state["map_zoom"] = SEARCH_RESULT_ZOOM
                st.session_state["search_marker"] = {
                    "lat": lat,
                    "lon": lon,
                    "display_name": selected_place.get("display_name", "Suchergebnis"),
                }

                st.rerun()

        st.divider()

        area_name = st.text_input("Bezeichnung des Bereichs", value=DEFAULT_AREA_NAME)

        st.subheader("Manuelle Bounding Box")
        south = st.number_input(
            "south / min latitude",
            value=DEFAULT_MANUAL_SOUTH,
            format="%.6f",
        )
        west = st.number_input(
            "west / min longitude",
            value=DEFAULT_MANUAL_WEST,
            format="%.6f",
        )
        north = st.number_input(
            "north / max latitude",
            value=DEFAULT_MANUAL_NORTH,
            format="%.6f",
        )
        east = st.number_input(
            "east / max longitude",
            value=DEFAULT_MANUAL_EAST,
            format="%.6f",
        )

        use_manual_bbox = st.checkbox("Manuelle Bounding Box verwenden", value=False)

    manual_selected_bounds: AreaBounds | None = None
    manual_bbox_error: Exception | None = None

    # Validate the manual bounding box before the map is rendered so it can
    # be displayed immediately on the map.
    if use_manual_bbox:
        try:
            manual_selected_bounds = AreaBounds(
                south=south,
                west=west,
                north=north,
                east=east,
            )
        except Exception as exc:
            manual_bbox_error = exc

    col_map, col_out = st.columns(MAP_COLUMN_RATIO)

    with col_map:
        st.subheader("Bereichsauswahl")

        map_center = st.session_state["map_center"]
        map_zoom = st.session_state["map_zoom"]

        if manual_selected_bounds is not None:
            map_center = (
                manual_selected_bounds.center_lat,
                manual_selected_bounds.center_lon,
            )

        fmap = make_base_map(
            center=map_center,
            zoom=map_zoom,
            search_marker=st.session_state.get("search_marker"),
            selected_bounds=manual_selected_bounds,
        )

        map_key = (
            f"base_map_"
            f"{map_center[0]:.6f}_"
            f"{map_center[1]:.6f}_"
            f"{map_zoom}_"
            f"{use_manual_bbox}"
        )

        map_data = st_folium(
            fmap,
            height=BASE_MAP_HEIGHT_PX,
            width=None,
            returned_objects=["last_active_drawing", "all_drawings"],
            key=map_key,
        )

    selected_bounds: AreaBounds | None = None

    if use_manual_bbox:
        selected_bounds = manual_selected_bounds

        if manual_bbox_error is not None:
            st.error(f"Ungültige Bounding Box: {manual_bbox_error}")
    else:
        selected_bounds = bounds_from_drawings(map_data)

    with col_out:
        st.subheader("Auswahl")

        if selected_bounds is not None:
            st.json(selected_bounds.model_dump())
            st.metric(
                "Fläche ca. km²",
                f"{selected_bounds.approx_area_km2():.{AREA_DISPLAY_DECIMALS}f}",
            )
        else:
            st.info("Noch kein Bereich ausgewählt. Zeichne ein Rechteck/Polygon auf der Karte.")

        build_clicked = st.button(
            "GridNetwork erzeugen",
            type="primary",
            disabled=selected_bounds is None,
            key="build_grid_network_button",
        )

        if build_clicked:
            if selected_bounds is None:
                st.error("Bitte zuerst einen Bereich auswählen.")
                return

            scenario = area_name.strip() or FALLBACK_SCENARIO_NAME

            with st.spinner("GridNetwork wird erzeugt ..."):
                try:
                    builder = OSMNetworkBuilder(
                        top=selected_bounds.north,
                        bottom=selected_bounds.south,
                        left=selected_bounds.west,
                        right=selected_bounds.east,
                        scenario=scenario,
                        conda_env=GRIDCREATOR_CONDA_ENV,
                    )

                    full_network = builder.build()

                    st.session_state["full_network"] = full_network
                    st.session_state["built_network"] = full_network
                    st.session_state["built_bounds"] = selected_bounds
                    st.session_state["built_scenario"] = scenario
                    st.session_state["max_households"] = INTERNAL_MAX_HOUSEHOLDS_PER_FEEDER

                    # Start with the complete generated network. Transformer
                    # filtering is applied only after explicit user selection.
                    st.session_state["selected_trafo_id"] = None

                    # Reset interactive configuration state for the new network.
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
    """Create all Streamlit session-state keys used by this page."""
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
        st.session_state["scenario_config_version"] = INITIAL_CONFIG_VERSION

    if "household_config_version" not in st.session_state:
        st.session_state["household_config_version"] = INITIAL_CONFIG_VERSION

    if "max_households" not in st.session_state:
        st.session_state["max_households"] = INTERNAL_MAX_HOUSEHOLDS_PER_FEEDER


def make_base_map(
    center: tuple[float, float] = DEFAULT_CENTER,
    zoom: int = DEFAULT_ZOOM,
    search_marker: dict[str, Any] | None = None,
    selected_bounds: AreaBounds | None = None,
) -> folium.Map:
    """Create the Folium map used for area selection."""
    fmap = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    if selected_bounds is not None:
        folium.Rectangle(
            bounds=[
                [selected_bounds.south, selected_bounds.west],
                [selected_bounds.north, selected_bounds.east],
            ],
            tooltip=MANUAL_BBOX_TOOLTIP,
            color=MANUAL_BBOX_COLOR,
            weight=MANUAL_BBOX_WEIGHT,
            fill=MANUAL_BBOX_FILL,
        ).add_to(fmap)

    if search_marker:
        folium.Marker(
            location=[search_marker["lat"], search_marker["lon"]],
            tooltip="Suchergebnis",
            popup=folium.Popup(
                search_marker["display_name"],
                max_width=SEARCH_MARKER_POPUP_MAX_WIDTH,
            ),
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


@st.cache_data(show_spinner=False, ttl=NOMINATIM_CACHE_TTL_SECONDS)
def search_place(query: str) -> list[dict[str, Any]]:
    """Search a German place name or address using Nominatim."""
    query = query.strip()

    if not query:
        return []

    response = requests.get(
        NOMINATIM_SEARCH_URL,
        params={
            "q": query,
            "format": "jsonv2",
            "limit": NOMINATIM_RESULT_LIMIT,
            "addressdetails": NOMINATIM_ADDRESS_DETAILS,
            "countrycodes": NOMINATIM_COUNTRY_CODES,
        },
        headers={
            "User-Agent": NOMINATIM_USER_AGENT,
        },
        timeout=NOMINATIM_REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()
    return response.json()


def format_search_result(result: dict[str, Any]) -> str:
    """Return the display label for one Nominatim search result."""
    return result.get("display_name", "Unbekanntes Suchergebnis")


def is_overpass_timeout_error(exc: Exception) -> bool:
    """Detect likely Overpass timeout or connection errors."""
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
    """Convert the current Folium drawing into AreaBounds."""
    if not map_data:
        return None

    drawings = map_data.get("all_drawings")

    # After "Clear all", all_drawings is empty while last_active_drawing may
    # still contain the old area. all_drawings is therefore the reliable state.
    if isinstance(drawings, list):
        if not drawings:
            return None

        drawing = drawings[DRAWING_LAST_ITEM_INDEX]
    else:
        drawing = map_data.get("last_active_drawing")

    if not drawing:
        return None

    geometry = drawing.get("geometry") or {}
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    if not coordinates:
        return None

    lon_lat_pairs: list[tuple[float, float]] = []

    if geom_type == "Polygon":
        lon_lat_pairs = [
            (float(lon), float(lat))
            for lon, lat in coordinates[DRAWING_FIRST_RING_INDEX]
        ]

    elif geom_type == "MultiPolygon":
        for polygon in coordinates:
            lon_lat_pairs.extend(
                (float(lon), float(lat))
                for lon, lat in polygon[DRAWING_FIRST_RING_INDEX]
            )

    if not lon_lat_pairs:
        return None

    lons = [point[0] for point in lon_lat_pairs]
    lats = [point[1] for point in lon_lat_pairs]

    return AreaBounds(
        south=min(lats),
        west=min(lons),
        north=max(lats),
        east=max(lons),
    )


def show_grid_model_result(network, selected_bounds: AreaBounds) -> None:
    """Show summary data and JSON export for the displayed GridNetwork."""
    st.subheader("Netzmodell")

    selected_trafo_id = st.session_state.get("selected_trafo_id")

    if selected_trafo_id is None:
        st.caption(
            "Es ist aktuell kein Trafo-Filter aktiv. "
            "Die Netzwerkkonfiguration zeigt das vollständig erzeugte GridNetwork "
            "für den ausgewählten Kartenbereich."
        )
    else:
        st.caption(
            "Aktiver Trafo-Filter: Es wird nur der aktuell ausgewählte "
            "Transformatorbereich angezeigt. Der JSON-Export enthält nur die "
            "zugehörigen Knoten, Leitungen, Transformatoren und Haushalte."
        )

    bus_count = len(network.buses)
    line_count = len(network.lines)
    transformer_count = len(network.transformers)
    household_count = len(network.household_bus_ids)
    devices = device_summary(network)

    c1, c2, c3, c4 = st.columns(GRID_METRIC_COLUMN_COUNT)
    c1.metric("Knoten", bus_count)
    c2.metric("Leitungen", line_count)
    c3.metric("Haushalte", household_count)
    c4.metric("Transformatoren", transformer_count)

    d1, d2, d3, d4 = st.columns(DEVICE_METRIC_COLUMN_COUNT)
    d1.metric("Haushalte mit Elektroauto", devices["ev_count"])
    d2.metric("Haushalte mit Wärmepumpe", devices["heat_pump_count"])
    d3.metric("Haushalte mit Batteriespeicher", devices["battery_count"])
    d4.metric("Haushalte mit PV-Anlage", devices["pv_count"])

    st.write("**Ausgewählter Kartenbereich**")
    st.json(
        {
            "bbox": selected_bounds.model_dump(),
            "area_km2": selected_bounds.approx_area_km2(),
        }
    )

    filtering_mode = (
        FILTERING_MODE_COMPLETE_NETWORK
        if selected_trafo_id is None
        else FILTERING_MODE_TRANSFORMER_AREA
    )

    st.write("**Erzeugte Netzwerkdaten**")
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
    """Standalone entry point for Streamlit."""
    st.set_page_config(page_title="GridKIT map_ui", layout="wide")
    render_map_ui()


if __name__ == "__main__":
    main()