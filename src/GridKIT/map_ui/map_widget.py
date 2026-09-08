# ─────────────────────────────────────────────────────────────
# map_ui/map_widget.py
#
# Streamlit + Folium GUI for selecting an area and displaying
# GridNetwork information from grid_model's OSMNetworkBuilder.
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
from map_ui.household_config import (
    bool_from_choice,
    build_household_configuration,
    choice_index_from_bool,
    default_scenario_assumptions,
    gridcreator_defaults,
    json_dumps_pretty,
)
from map_ui.network_visualization import show_network_visualization
from map_ui.osm_fetcher import AreaBounds


DEFAULT_CENTER = (49.0069, 8.4037)  # Karlsruhe
DEFAULT_ZOOM = 15

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "GridKIT-map-ui/0.1"


def render_map_ui() -> None:
    """Page body — no st.set_page_config here, so this can be composed as one
    page of a larger app (see scripts/app.py) as well as run standalone
    (see main() below, which owns page config for the standalone case)."""
    st.title("GridKIT Karte")
    st.caption(
        "Ort suchen → Bereich auswählen → GridNetwork erzeugen → Netzansicht visualisieren → "
        "Haushalte konfigurieren → speichern & trainieren"
    )

    if "built_network" not in st.session_state:
        st.session_state["built_network"] = None
    if "built_bounds" not in st.session_state:
        st.session_state["built_bounds"] = None
    if "built_scenario" not in st.session_state:
        st.session_state["built_scenario"] = None

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

    if "scenario_config_version" not in st.session_state:
        st.session_state["scenario_config_version"] = 0

    if "household_config_version" not in st.session_state:
        st.session_state["household_config_version"] = 0

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

                    network = builder.build()

                    st.session_state["built_network"] = network
                    st.session_state["built_bounds"] = selected_bounds
                    st.session_state["built_scenario"] = scenario
                    st.session_state["max_households"] = int(max_households)

                    # Alte individuelle Haushalt-Anpassungen zurücksetzen,
                    # wenn ein neues GridNetwork erzeugt wird.
                    st.session_state["household_overrides"] = {}
                    st.session_state["household_config_version"] += 1

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

    built_network = st.session_state.get("built_network")
    built_bounds = st.session_state.get("built_bounds")

    if built_network is not None and built_bounds is not None:
        st.divider()
        show_grid_model_result(built_network, built_bounds)
        st.divider()
        show_network_visualization(built_network, built_bounds)
        st.divider()
        household_configuration = show_household_configuration(built_network, built_bounds)
        if household_configuration is not None:
            st.divider()
            show_training_section(built_network, household_configuration)


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
    st.subheader("GridNetwork aus grid_model")

    bus_count = len(network.buses)
    line_count = len(network.lines)
    transformer_count = len(network.transformers)
    household_count = len(network.household_bus_ids)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buses / Knoten", bus_count)
    c2.metric("Lines / Kanten", line_count)
    c3.metric("Households", household_count)
    c4.metric("Transformers", transformer_count)

    st.write("**Ausgewählter Kartenbereich**")
    st.json(
        {
            "bbox": selected_bounds.model_dump(),
            "area_km2": selected_bounds.approx_area_km2(),
        }
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


def show_household_configuration(network, selected_bounds: AreaBounds) -> None:
    st.subheader("Haushaltskonfiguration")

    household_ids = sorted(str(bus_id) for bus_id in getattr(network, "household_bus_ids", []))

    if not household_ids:
        st.warning(
            "Im erzeugten GridNetwork wurden keine Haushalts-/Last-Busse gefunden. "
            "Die Haushaltskonfiguration kann deshalb nicht angewendet werden."
        )
        return

    if st.session_state.pop("scenario_assumptions_saved_message", False):
        st.success("Szenario-Annahmen wurden gespeichert.")

    if st.session_state.pop("scenario_assumptions_reset_message", False):
        st.success("Szenario-Annahmen wurden zurückgesetzt.")

    saved_household = st.session_state.pop("household_saved_message", None)
    if saved_household:
        st.success(f"Individuelle Anpassung für {saved_household} gespeichert.")

    reset_household = st.session_state.pop("household_reset_message", None)
    if reset_household:
        st.info(f"Individuelle Anpassung für {reset_household} wurde gelöscht.")

    st.session_state.setdefault("household_overrides", {})
    st.session_state.setdefault("scenario_assumptions", default_scenario_assumptions())
    st.session_state.setdefault("scenario_config_version", 0)
    st.session_state.setdefault("household_config_version", 0)

    household_overrides: dict[str, dict[str, Any]] = st.session_state["household_overrides"]
    saved_assumptions: dict[str, Any] = st.session_state["scenario_assumptions"]

    uses_gridcreator_defaults = bool(gridcreator_defaults(network))

    st.markdown("### Szenario-Annahmen für das Gebiet")

    if uses_gridcreator_defaults:
        st.info(
            "Für dieses Netz liegen reale GridCreator-Gerätedaten vor (EV/Wärmepumpe/"
            "Batterie/PV). Diese werden automatisch als Standardwerte je Haushalt "
            "verwendet — die Szenario-Anteile unten sind für dieses Netz inaktiv. "
            "Einzelne Haushalte können weiterhin unten manuell angepasst werden."
        )
    else:
        st.info(
            "Für dieses Netz liegen keine realen GridCreator-Gerätedaten vor "
            "(z. B. ding0-Direktimport oder Stub-Netz). EV/Wärmepumpe folgen daher "
            "den Szenario-Anteilen unten; Batterie/PV sind ohne manuelle Anpassung "
            "standardmäßig aus."
        )

    st.caption(
        "Die folgenden Eingabefelder sind zunächst ein Entwurf. "
        "Mit „Szenario-Annahmen speichern“ werden sie für die JSON-Konfiguration übernommen."
    )

    widget_suffix = st.session_state["scenario_config_version"]

    high_col1, high_col2, high_col3, high_col4 = st.columns(4)

    draft_ev_share_percent = high_col1.slider(
        "Anteil Haushalte mit EV (%)",
        min_value=0,
        max_value=100,
        value=int(saved_assumptions["ev_share_percent"]),
        step=5,
        help="Annahme für den Anteil der Haushalte, die ein Elektrofahrzeug besitzen sollen.",
        key=f"draft_ev_share_percent_{widget_suffix}",
        disabled=uses_gridcreator_defaults,
    )

    draft_heat_pump_share_percent = high_col2.slider(
        "Anteil Haushalte mit WP (%)",
        min_value=0,
        max_value=100,
        value=int(saved_assumptions["heat_pump_share_percent"]),
        step=5,
        help="Annahme für den Anteil der Haushalte, die eine Wärmepumpe besitzen sollen.",
        key=f"draft_heat_pump_share_percent_{widget_suffix}",
        disabled=uses_gridcreator_defaults,
    )

    draft_global_load_scaling_factor = high_col3.number_input(
        "Verbrauchsfaktor global",
        min_value=0.1,
        max_value=5.0,
        value=float(saved_assumptions["global_load_scaling_factor"]),
        step=0.1,
        format="%.2f",
        help="1.0 bedeutet unverändert, 1.2 bedeutet 20 % höherer Verbrauch.",
        key=f"draft_global_load_scaling_factor_{widget_suffix}",
    )

    draft_selection_seed = high_col4.number_input(
        "Zufallswert für Haushalt-Auswahl",
        min_value=0,
        max_value=99999,
        value=int(saved_assumptions["selection_seed"]),
        step=1,
        help="Sorgt dafür, dass die prozentuale Auswahl reproduzierbar bleibt.",
        key=f"draft_selection_seed_{widget_suffix}",
    )

    draft_assumptions = {
        "ev_share_percent": int(draft_ev_share_percent),
        "heat_pump_share_percent": int(draft_heat_pump_share_percent),
        "global_load_scaling_factor": float(draft_global_load_scaling_factor),
        "selection_seed": int(draft_selection_seed),
    }

    high_action_col1, high_action_col2 = st.columns(2)

    if high_action_col1.button("Szenario-Annahmen speichern"):
        st.session_state["scenario_assumptions"] = draft_assumptions
        st.session_state["scenario_config_version"] += 1
        st.session_state["scenario_assumptions_saved_message"] = True
        st.rerun()

    if high_action_col2.button("Szenario-Annahmen zurücksetzen"):
        st.session_state["scenario_assumptions"] = default_scenario_assumptions()
        st.session_state["scenario_config_version"] += 1
        st.session_state["scenario_assumptions_reset_message"] = True
        st.rerun()

    if draft_assumptions != saved_assumptions:
        st.warning(
            "Es gibt nicht gespeicherte Änderungen in den Szenario-Annahmen. "
            "Klicke auf „Szenario-Annahmen speichern“, damit diese Werte in die JSON-Konfiguration übernommen werden."
        )

    saved_ev_share_percent = int(saved_assumptions["ev_share_percent"])
    saved_heat_pump_share_percent = int(saved_assumptions["heat_pump_share_percent"])
    saved_global_load_scaling_factor = float(saved_assumptions["global_load_scaling_factor"])
    saved_selection_seed = int(saved_assumptions["selection_seed"])

    st.markdown("### Individuelle Anpassung einzelner Haushalte")

    selected_household = st.selectbox(
        "Haushalt / Lastpunkt auswählen",
        household_ids,
        help="Hier kann ein konkreter vorhandener Haushalts-/Last-Bus individuell angepasst werden.",
    )

    current_override = household_overrides.get(selected_household, {})
    household_widget_suffix = st.session_state["household_config_version"]

    def _auto_help(device_label: str, share_percent: int | None) -> str:
        if uses_gridcreator_defaults:
            return f"Automatisch = Zuordnung aus GridCreator für diesen Haushalt ({device_label})."
        if share_percent is not None:
            return f"Automatisch = {share_percent}% Szenario-Anteil ({device_label}), zufällig verteilt."
        return f"Automatisch = aus (kein Szenario-Anteil für {device_label})."

    low_col1, low_col2, low_col3, low_col4, low_col5 = st.columns(5)

    ev_choice = low_col1.selectbox(
        "EV für diesen Haushalt",
        options=["Automatisch (Standard)", "Ja", "Nein"],
        index=choice_index_from_bool(current_override.get("has_ev")),
        key=f"ev_choice_{selected_household}_{household_widget_suffix}",
        help=_auto_help("EV", saved_ev_share_percent),
    )

    heat_pump_choice = low_col2.selectbox(
        "WP für diesen Haushalt",
        options=["Automatisch (Standard)", "Ja", "Nein"],
        index=choice_index_from_bool(current_override.get("has_heat_pump")),
        key=f"heat_pump_choice_{selected_household}_{household_widget_suffix}",
        help=_auto_help("Wärmepumpe", saved_heat_pump_share_percent),
    )

    battery_choice = low_col3.selectbox(
        "Batterie für diesen Haushalt",
        options=["Automatisch (Standard)", "Ja", "Nein"],
        index=choice_index_from_bool(current_override.get("has_battery")),
        key=f"battery_choice_{selected_household}_{household_widget_suffix}",
        help=_auto_help("Batterie", None),
    )

    pv_choice = low_col4.selectbox(
        "PV für diesen Haushalt",
        options=["Automatisch (Standard)", "Ja", "Nein"],
        index=choice_index_from_bool(current_override.get("has_pv")),
        key=f"pv_choice_{selected_household}_{household_widget_suffix}",
        help=_auto_help("PV", None),
    )

    load_factor_mode = low_col5.selectbox(
        "Verbrauchsfaktor-Modus",
        options=["Automatisch aus Szenario-Annahmen", "Individuell festlegen"],
        index=1 if "load_scaling_factor" in current_override else 0,
        key=f"load_factor_mode_{selected_household}_{household_widget_suffix}",
        help=(
            "Automatisch bedeutet: Der globale Verbrauchsfaktor aus den Szenario-Annahmen gilt. "
            "Individuell bedeutet: Für diesen Haushalt wird ein eigener Verbrauchsfaktor gespeichert."
        ),
    )

    individual_load_scaling_factor = low_col5.number_input(
        "Individueller Verbrauchsfaktor",
        min_value=0.1,
        max_value=5.0,
        value=float(current_override.get("load_scaling_factor", saved_global_load_scaling_factor)),
        step=0.1,
        format="%.2f",
        key=f"load_factor_{selected_household}_{household_widget_suffix}",
        disabled=load_factor_mode == "Automatisch aus Szenario-Annahmen",
        help="Dieser Wert wird nur gespeichert, wenn der Modus auf „Individuell festlegen“ steht.",
    )

    action_col1, action_col2 = st.columns(2)

    if action_col1.button(
        "Individuelle Anpassung für diesen Haushalt speichern",
        key=f"save_override_{selected_household}",
    ):
        new_override: dict[str, Any] = {}

        ev_override = bool_from_choice(ev_choice)
        heat_pump_override = bool_from_choice(heat_pump_choice)
        battery_override = bool_from_choice(battery_choice)
        pv_override = bool_from_choice(pv_choice)

        if ev_override is not None:
            new_override["has_ev"] = ev_override

        if heat_pump_override is not None:
            new_override["has_heat_pump"] = heat_pump_override

        if battery_override is not None:
            new_override["has_battery"] = battery_override

        if pv_override is not None:
            new_override["has_pv"] = pv_override

        if load_factor_mode == "Individuell festlegen":
            new_override["load_scaling_factor"] = float(individual_load_scaling_factor)

        if new_override:
            household_overrides[selected_household] = new_override
            st.session_state["household_config_version"] += 1
            st.session_state["household_saved_message"] = selected_household
            st.rerun()
        else:
            household_overrides.pop(selected_household, None)
            st.session_state["household_config_version"] += 1
            st.session_state["household_reset_message"] = selected_household
            st.rerun()

    if action_col2.button(
        "Individuelle Anpassung löschen",
        key=f"delete_override_{selected_household}",
    ):
        household_overrides.pop(selected_household, None)
        st.session_state["household_config_version"] += 1
        st.session_state["household_reset_message"] = selected_household
        st.rerun()

    household_configuration = build_household_configuration(
        network=network,
        selected_bounds=selected_bounds,
        household_ids=household_ids,
        ev_share_percent=saved_ev_share_percent,
        heat_pump_share_percent=saved_heat_pump_share_percent,
        global_load_scaling_factor=saved_global_load_scaling_factor,
        selection_seed=saved_selection_seed,
        household_overrides=household_overrides,
    )

    st.markdown("### Zusammenfassung der aktuellen Haushaltskonfiguration")

    resolved = household_configuration["resolved"]

    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5, summary_col6 = st.columns(6)
    summary_col1.metric("Haushalte gesamt", len(household_ids))
    summary_col2.metric("Haushalte mit EV", len(resolved["ev_bus_ids"]))
    summary_col3.metric("Haushalte mit WP", len(resolved["heat_pump_bus_ids"]))
    summary_col4.metric("Haushalte mit Batterie", len(resolved["battery_bus_ids"]))
    summary_col5.metric("Haushalte mit PV", len(resolved["pv_bus_ids"]))
    summary_col6.metric("Individuelle Anpassungen", len(household_overrides))

    with st.expander("household_configuration.json anzeigen"):
        st.json(household_configuration)

    st.download_button(
        label="household_configuration.json herunterladen",
        data=json_dumps_pretty(household_configuration),
        file_name="household_configuration.json",
        mime="application/json",
    )

    return household_configuration


def show_training_section(network, household_configuration: dict[str, Any]) -> None:
    """Save the current network + household configuration, and optionally
    launch a training run in the background. Progress/results are viewed in
    the separate dashboard app (streamlit run src/GridKIT/dashboard/app.py),
    not here — this section only creates/launches runs.
    """
    import core.constants as const
    from core import run_store as rs

    st.subheader("Speichern & Training")

    saved_run_id = st.session_state.pop("last_saved_run_id", None)
    if saved_run_id:
        st.success(f"Gespeichert als **{saved_run_id}**.")
    launched_run_id = st.session_state.pop("last_launched_run_id", None)
    if launched_run_id:
        st.success(
            f"Training für **{launched_run_id}** gestartet. Fortschritt und Ergebnisse siehst du im "
            f"Dashboard (`streamlit run src/GridKIT/dashboard/app.py`)."
        )

    st.caption(
        "Speichert dieses Netz mit der aktuellen Haushaltskonfiguration unter `runs/`, sodass mehrere "
        "Netze parallel gespeichert werden können. Trainings-Details (Iterationen, Gewichtung o. Ä.) "
        "werden bewusst nicht angezeigt — ein Lauf verwendet immer dieselben, festen Einstellungen."
    )

    n_households = len(getattr(network, "household_bus_ids", []))
    default_name = f"{getattr(network, 'area_name', None) or network.network_id} ({n_households} Haushalte)"
    run_name = st.text_input(
        "Name für diesen Lauf",
        value=default_name,
        help="Nur ein Anzeigename, um mehrere gespeicherte Netze auseinanderzuhalten — muss nicht eindeutig sein.",
    )

    # Each training run spawns its own Ray instance + worker processes — running
    # several at once has been observed to exhaust memory and crash Ray's
    # actors (surfacing as cryptic RLlib errors in the dashboard, or a run
    # silently orphaned). `stale` running entries (no status update in a long
    # time — see core.run_store.STALE_AFTER_SECONDS) are excluded: those are
    # themselves almost certainly dead, not a real second training in progress.
    active_runs = [r for r in rs.list_runs() if r["state"] == rs.RUNNING and not r.get("stale")]
    if active_runs:
        names = ", ".join(f"**{r.get('name') or r['run_id']}**" for r in active_runs)
        st.warning(
            f"Es läuft bereits ein Training ({names}). Mehrere gleichzeitige Trainings können den "
            "Rechner überlasten und Abstürze verursachen — bitte warten, bis es fertig ist (Fortschritt "
            "im Dashboard), bevor ein weiteres gestartet wird. Speichern allein ist weiterhin möglich."
        )

    col1, col2 = st.columns(2)
    save_only_clicked = col1.button("Nur speichern")
    save_and_train_clicked = col2.button(
        "Speichern & Training starten", type="primary", disabled=bool(active_runs),
    )

    if save_only_clicked:
        run_id = rs.save_network_only(run_name, network, household_configuration, network_source="map_ui")
        st.session_state["last_saved_run_id"] = run_id
        st.rerun()

    if save_and_train_clicked:
        run_id = rs.create_run(
            run_name, network, household_configuration,
            iterations=const.PIPELINE_TRAINING_ITERATIONS,
            seeds=const.PIPELINE_EVALUATION_SEEDS,
            network_source="map_ui",
        )
        try:
            _launch_training(run_id)
        except Exception as exc:
            rs.set_status(run_id, state=rs.FAILED, message=f"Start fehlgeschlagen: {exc}")
            st.error("Training konnte nicht gestartet werden.")
            st.exception(exc)
        else:
            st.session_state["last_launched_run_id"] = run_id
            st.rerun()


def _launch_training(run_id: str) -> None:
    """Spawn a detached scripts.train_run subprocess writing into runs/<run_id>/."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    from core import run_store as rs

    script_dir = Path(__file__).resolve().parent.parent   # src/GridKIT/
    src_dir = script_dir.parent                             # src/
    repo_root = src_dir.parent                               # repo root

    run_directory = rs.run_dir(run_id, root=repo_root / "runs")
    env = {**os.environ, "PYTHONPATH": f"{script_dir}{os.pathsep}{src_dir}"}
    with open(run_directory / "train.log", "w") as logf:
        subprocess.Popen(
            [sys.executable, "-m", "GridKIT.scripts.train_run", "--run-dir", str(run_directory)],
            cwd=str(repo_root), env=env, stdout=logf, stderr=subprocess.STDOUT,
        )


def main() -> None:
    """Standalone entry point: streamlit run src/GridKIT/map_ui/map_widget.py"""
    st.set_page_config(page_title="GridKIT map_ui", layout="wide")
    render_map_ui()


if __name__ == "__main__":
    main()