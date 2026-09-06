# ─────────────────────────────────────────────────────────────
# map_ui/map_widget.py
#
# Streamlit + Folium map view WITH network configuration:
#   Ort suchen → Bereich auswählen → GridNetwork erzeugen →
#   Geräte-Szenario je Haushalt konfigurieren.
#
# render_map_config() is the embeddable body (no page config/title) so the
# dashboard can use THIS map view instead of maintaining its own. main() is the
# thin standalone wrapper.
#
# The network builder is injected (`build_network`) so this module stays free of
# grid_model imports: the dashboard passes its ding0→OSM builder, standalone use
# falls back to the plain OSM builder in map_ui.osm_fetcher.
#
# Run standalone:
#   python -m streamlit run src/GridKIT/map_ui/map_widget.py
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any, Callable

import folium
import requests
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

import core.constants as const
from core.models import GridNetwork
from map_ui.household_config import (
    bool_from_choice,
    build_household_configuration,
    choice_index_from_bool,
    default_scenario_assumptions,
    device_layout_from_configuration,
    json_dumps_pretty,
)
from map_ui.osm_fetcher import AreaBounds, OsmFetchConfig, build_grid_network_from_bounds

DEFAULT_CENTER = (49.0069, 8.4037)  # Karlsruhe
DEFAULT_ZOOM = 15

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "GridKIT-map-ui/0.1"

DEVICE_LABELS_DE = {
    const.DEVICE_EV: "EV",
    const.DEVICE_BATTERY: "Batterie",
    const.DEVICE_HEAT_PUMP: "Wärmepumpe",
    const.DEVICE_PV: "PV",
}

# (device, assumption field, slider label) — one row per configurable device share.
_SHARE_FIELDS = (
    (const.DEVICE_EV, "ev_share_percent", "Anteil Haushalte mit EV (%)"),
    (const.DEVICE_BATTERY, "battery_share_percent", "Anteil Haushalte mit Batterie (%)"),
    (const.DEVICE_HEAT_PUMP, "heat_pump_share_percent", "Anteil Haushalte mit WP (%)"),
    (const.DEVICE_PV, "pv_share_percent", "Anteil Haushalte mit PV (%)"),
)

_OVERRIDE_FIELDS = (
    (const.DEVICE_EV, "has_ev"),
    (const.DEVICE_BATTERY, "has_battery"),
    (const.DEVICE_HEAT_PUMP, "has_heat_pump"),
    (const.DEVICE_PV, "has_pv"),
)

_AUTO_CHOICE = "Automatisch aus Szenario-Annahmen"


def default_osm_build(bounds: AreaBounds, area_name: str) -> GridNetwork:
    """Plain OSM/Overpass build — the standalone fallback when no builder is injected."""
    result = build_grid_network_from_bounds(bounds, area_name=area_name, config=OsmFetchConfig())
    for warning in result.warnings:
        st.warning(warning)
    return result.grid_network


def render_map_config(
    *,
    build_network: Callable[[AreaBounds, str], GridNetwork] | None = None,
    on_network_built: Callable[[GridNetwork], None] | None = None,
) -> None:  # pragma: no cover (UI)
    """Map view + network configuration, embeddable as a section of another page.

    `build_network(bounds, area_name)` produces the GridNetwork (injected so the
    dashboard can use ding0 with an OSM fallback). `on_network_built(network)` lets
    the host wire the result into its own session state. The resolved device layout
    is written to st.session_state["layout"] as {bus_id: HouseholdDevices}.
    """
    build_network = build_network or default_osm_build
    ss = st.session_state

    ss.setdefault("map_center", DEFAULT_CENTER)
    ss.setdefault("map_zoom", DEFAULT_ZOOM)
    ss.setdefault("search_results", [])
    ss.setdefault("search_marker", None)
    ss.setdefault("built_bounds", None)
    ss.setdefault("household_overrides", {})
    ss.setdefault("scenario_assumptions", default_scenario_assumptions())
    ss.setdefault("scenario_config_version", 0)
    ss.setdefault("household_config_version", 0)

    area_name, build_clicked = _render_sidebar_inputs()
    selected_bounds = _render_map_and_selection()

    if build_clicked:
        if not selected_bounds:
            st.error("Bitte zuerst einen Bereich auswählen — Rechteck auf der Karte zeichnen.")
        else:
            scenario = area_name.strip() or "selected_area"
            try:
                network = build_network(selected_bounds, scenario)
            except Exception as exc:
                st.error("Das GridNetwork konnte nicht erzeugt werden.")
                with st.expander("Technische Details anzeigen"):
                    st.exception(exc)
                return

            ss["built_bounds"] = selected_bounds
            ss["household_overrides"] = {}
            ss["household_config_version"] += 1
            if on_network_built:
                on_network_built(network)
            else:
                ss["network"] = network
            st.success(f"GridNetwork erzeugt — {len(network.household_bus_ids)} Haushalte.")

    network = ss.get("network")
    built_bounds = ss.get("built_bounds")
    if network is not None and built_bounds is not None:
        st.divider()
        _render_network_summary(network, built_bounds)
        st.divider()
        render_household_configuration(network, built_bounds)


def _render_sidebar_inputs() -> tuple[str, bool]:  # pragma: no cover (UI)
    ss = st.session_state
    with st.sidebar:
        st.subheader("Ort suchen")
        place_query = st.text_input(
            "Ort, Stadt, Straße oder Adresse",
            placeholder="z. B. Karlsruhe, Kaiserstraße Karlsruhe",
            key="map_place_query",
        )
        if st.button("Ort suchen", key="map_search_button"):
            try:
                ss["search_results"] = search_place(place_query)
                if not ss["search_results"]:
                    st.warning("Keine Suchergebnisse gefunden.")
            except Exception as exc:
                st.error("Die Ortssuche ist fehlgeschlagen.")
                st.exception(exc)
                ss["search_results"] = []

        results = ss.get("search_results", [])
        if results:
            idx = st.selectbox(
                "Suchergebnis auswählen",
                options=list(range(len(results))),
                format_func=lambda i: format_search_result(results[i]),
                key="map_search_choice",
            )
            if st.button("Karte auf Suchergebnis zentrieren", key="map_center_button"):
                place = results[idx]
                ss["map_center"] = (float(place["lat"]), float(place["lon"]))
                ss["map_zoom"] = 16
                ss["search_marker"] = {
                    "lat": float(place["lat"]),
                    "lon": float(place["lon"]),
                    "display_name": place.get("display_name", "Suchergebnis"),
                }
                st.rerun()

        area_name = st.text_input("Gebietsname", value="selected_area", key="map_area_name")
        build_clicked = st.button("GridNetwork erzeugen", type="primary", key="map_build_button")

    return area_name, build_clicked


def _render_map_and_selection() -> AreaBounds | None:  # pragma: no cover (UI)
    ss = st.session_state
    st.caption(
        "Ort suchen (Seitenleiste) → Rechteck/Polygon auf der Karte zeichnen → "
        "**GridNetwork erzeugen**."
    )
    fmap = make_base_map(
        center=ss["map_center"],
        zoom=ss["map_zoom"],
        search_marker=ss.get("search_marker"),
    )
    map_key = f"base_map_{ss['map_center'][0]:.6f}_{ss['map_center'][1]:.6f}_{ss['map_zoom']}"
    map_data = st_folium(
        fmap, height=560, width=None,
        returned_objects=["last_active_drawing", "all_drawings"], key=map_key,
    )

    bounds = bounds_from_drawings(map_data)
    if bounds:
        st.caption(f"Ausgewählter Bereich ≈ **{bounds.approx_area_km2():.2f} km²**.")
    else:
        st.info("Noch kein Bereich ausgewählt. Zeichne ein Rechteck/Polygon auf der Karte.")
    return bounds


def _render_network_summary(network, selected_bounds: AreaBounds) -> None:  # pragma: no cover (UI)
    st.subheader("Erzeugtes GridNetwork")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Knoten", len(network.buses))
    c2.metric("Leitungen", len(network.lines))
    c3.metric("Haushalte", len(network.household_bus_ids))
    c4.metric("Transformatoren", len(network.transformers))

    grid_json = network.model_dump_json(indent=2)
    st.download_button(
        "grid_network.json herunterladen", data=grid_json,
        file_name="grid_network.json", mime="application/json", key="download_grid_network",
    )


def render_household_configuration(network, selected_bounds: AreaBounds) -> None:  # pragma: no cover (UI)
    """Scenario shares + per-household overrides → st.session_state['layout']."""
    ss = st.session_state
    st.subheader("Netzwerkkonfiguration — Geräte je Haushalt")

    household_ids = sorted(str(b) for b in getattr(network, "household_bus_ids", []))
    if not household_ids:
        st.warning("Im erzeugten GridNetwork wurden keine Haushalts-/Last-Busse gefunden.")
        return

    overrides: dict[str, dict[str, Any]] = ss["household_overrides"]
    saved: dict[str, Any] = ss["scenario_assumptions"]

    st.markdown("### Szenario-Annahmen für das Gebiet")
    st.caption(
        "Diese Anteile werden nicht aus OSM erkannt, sondern als Szenario-Annahme gesetzt. "
        "Die Auswahl ist über den Zufallswert reproduzierbar."
    )

    suffix = ss["scenario_config_version"]
    cols = st.columns(4)
    draft: dict[str, Any] = {}
    for col, (device, field, label) in zip(cols, _SHARE_FIELDS):
        draft[field] = int(col.slider(
            label, min_value=0, max_value=100, value=int(saved[field]), step=5,
            key=f"draft_{field}_{suffix}",
        ))

    c1, c2 = st.columns(2)
    draft["global_load_scaling_factor"] = float(c1.number_input(
        "Verbrauchsfaktor global", min_value=0.1, max_value=5.0,
        value=float(saved["global_load_scaling_factor"]), step=0.1, format="%.2f",
        help="1.0 bedeutet unverändert, 1.2 bedeutet 20 % höherer Verbrauch.",
        key=f"draft_load_scaling_{suffix}",
    ))
    draft["selection_seed"] = int(c2.number_input(
        "Zufallswert für Haushalt-Auswahl", min_value=0, max_value=99999,
        value=int(saved["selection_seed"]), step=1,
        key=f"draft_seed_{suffix}",
    ))

    a1, a2 = st.columns(2)
    if a1.button("Szenario-Annahmen anwenden", type="primary", key="apply_assumptions"):
        ss["scenario_assumptions"] = draft
        ss["scenario_config_version"] += 1
        st.rerun()
    if a2.button("Zurücksetzen", key="reset_assumptions"):
        ss["scenario_assumptions"] = default_scenario_assumptions()
        ss["scenario_config_version"] += 1
        st.rerun()

    if draft != saved:
        st.warning("Nicht angewendete Änderungen — auf **Szenario-Annahmen anwenden** klicken.")

    st.markdown("### Individuelle Anpassung einzelner Haushalte")
    selected = st.selectbox("Haushalt auswählen", household_ids, key="override_household")
    current = overrides.get(selected, {})
    hh_suffix = ss["household_config_version"]

    choice_cols = st.columns(4)
    choices: dict[str, str] = {}
    for col, (device, override_field) in zip(choice_cols, _OVERRIDE_FIELDS):
        choices[override_field] = col.selectbox(
            DEVICE_LABELS_DE[device], options=[_AUTO_CHOICE, "Ja", "Nein"],
            index=choice_index_from_bool(current.get(override_field)),
            key=f"choice_{override_field}_{selected}_{hh_suffix}",
        )

    o1, o2 = st.columns(2)
    if o1.button("Anpassung speichern", key=f"save_override_{selected}"):
        new_override = {
            field: value for field, value in
            ((f, bool_from_choice(c)) for f, c in choices.items())
            if value is not None
        }
        if new_override:
            overrides[selected] = new_override
        else:
            overrides.pop(selected, None)
        ss["household_config_version"] += 1
        st.rerun()
    if o2.button("Anpassung löschen", key=f"delete_override_{selected}"):
        overrides.pop(selected, None)
        ss["household_config_version"] += 1
        st.rerun()

    configuration = build_household_configuration(
        network=network,
        selected_bounds=selected_bounds,
        household_ids=household_ids,
        ev_share_percent=saved["ev_share_percent"],
        battery_share_percent=saved["battery_share_percent"],
        heat_pump_share_percent=saved["heat_pump_share_percent"],
        pv_share_percent=saved["pv_share_percent"],
        global_load_scaling_factor=saved["global_load_scaling_factor"],
        selection_seed=saved["selection_seed"],
        household_overrides=overrides,
    )

    # The layout is what the simulation actually consumes — the JSON is the exportable
    # description of the same decision.
    ss["layout"] = device_layout_from_configuration(household_ids, configuration)

    resolved = configuration["resolved"]
    st.markdown("### Zusammenfassung")
    s = st.columns(5)
    s[0].metric("Haushalte", len(household_ids))
    s[1].metric("EV", len(resolved["ev_bus_ids"]))
    s[2].metric("Batterie", len(resolved["battery_bus_ids"]))
    s[3].metric("Wärmepumpe", len(resolved["heat_pump_bus_ids"]))
    s[4].metric("PV", len(resolved["pv_bus_ids"]))

    with st.expander("household_configuration.json anzeigen"):
        st.json(configuration)
    st.download_button(
        "household_configuration.json herunterladen",
        data=json_dumps_pretty(configuration),
        file_name="household_configuration.json",
        mime="application/json",
        key="download_household_config",
    )


def make_base_map(
    center: tuple[float, float] = DEFAULT_CENTER,
    zoom: int = DEFAULT_ZOOM,
    search_marker: dict[str, Any] | None = None,
) -> folium.Map:
    fmap = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)

    if search_marker:
        folium.Marker(
            location=[search_marker["lat"], search_marker["lon"]],
            tooltip="Suchergebnis",
            popup=folium.Popup(search_marker["display_name"], max_width=350),
        ).add_to(fmap)

    Draw(
        export=False,
        draw_options={"polyline": False, "circle": False, "circlemarker": False,
                      "marker": False, "rectangle": True, "polygon": True},
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
        params={"q": query, "format": "jsonv2", "limit": 5,
                "addressdetails": 1, "countrycodes": "de"},
        headers={"User-Agent": NOMINATIM_USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def format_search_result(result: dict[str, Any]) -> str:
    return result.get("display_name", "Unbekanntes Suchergebnis")


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
        # GeoJSON polygon: [[[lon, lat], ...]]
        lon_lat_pairs = [(float(lon), float(lat)) for lon, lat in coordinates[0]]
    elif geom_type == "MultiPolygon":
        for polygon in coordinates:
            lon_lat_pairs.extend((float(lon), float(lat)) for lon, lat in polygon[0])

    if not lon_lat_pairs:
        return None

    lons = [p[0] for p in lon_lat_pairs]
    lats = [p[1] for p in lon_lat_pairs]

    return AreaBounds(south=min(lats), west=min(lons), north=max(lats), east=max(lons))


def main() -> None:  # pragma: no cover (UI)
    st.set_page_config(page_title="GridKIT map_ui", layout="wide")
    st.title("GridKIT Karte")
    render_map_config()


if __name__ == "__main__":
    main()
