\
# ─────────────────────────────────────────────────────────────
# map_ui/map_widget.py
#
# Streamlit + Folium GUI for selecting an OSM area and exporting
# core.models.GridNetwork to grid_model.
#
# Run:
#   streamlit run map_ui/map_widget.py
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
from typing import Any

import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from map_ui.osm_fetcher import AreaBounds, OsmFetchConfig, build_grid_network_from_bounds


DEFAULT_CENTER = (49.0069, 8.4037)  # Karlsruhe
DEFAULT_ZOOM = 15


def main() -> None:
    st.set_page_config(page_title="GridKIT map_ui", layout="wide")

    st.title("GridKIT map_ui")
    st.caption(
        "Kartenansicht → Bereich auswählen → Overpass API → "
        "GridNetwork JSON für grid_model"
    )

    with st.sidebar:
        st.header("Eingabe")
        st.write("Zeichne links ein Rechteck/Polygon oder gib unten eine Bounding Box ein.")

        area_name = st.text_input("area_name", value="selected_area")

        st.subheader("Manuelle Bounding Box")
        south = st.number_input("south / min latitude", value=49.0000, format="%.6f")
        west = st.number_input("west / min longitude", value=8.3900, format="%.6f")
        north = st.number_input("north / max latitude", value=49.0100, format="%.6f")
        east = st.number_input("east / max longitude", value=8.4100, format="%.6f")

        use_manual_bbox = st.checkbox("Manuelle Bounding Box verwenden", value=False)

        st.subheader("Overpass")
        max_area = st.number_input("max_area_km²", value=4.0, min_value=0.1, max_value=25.0, step=0.1)
        timeout = st.number_input("timeout_seconds", value=45, min_value=10, max_value=180, step=5)
        allow_fallback = st.checkbox("Highway-Fallback erlauben, falls keine power lines existieren", value=True)

        build_clicked = st.button("GridNetwork erzeugen", type="primary")

    col_map, col_out = st.columns([3, 2])

    with col_map:
        fmap = make_base_map()
        map_data = st_folium(fmap, height=650, width=None, returned_objects=["last_active_drawing", "all_drawings"])

    selected_bounds: AreaBounds | None = None

    if use_manual_bbox:
        selected_bounds = AreaBounds(south=south, west=west, north=north, east=east)
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

            config = OsmFetchConfig(
                timeout_seconds=int(timeout),
                max_area_km2=float(max_area),
                allow_highway_fallback=allow_fallback,
            )

            with st.spinner("Overpass API wird abgefragt und GridNetwork wird gebaut ..."):
                try:
                    result = build_grid_network_from_bounds(
                        selected_bounds,
                        area_name=area_name,
                        config=config,
                    )
                except Exception as exc:
                    st.exception(exc)
                    return

            st.success("GridNetwork erzeugt.")
            show_result(result)


def make_base_map() -> folium.Map:
    fmap = folium.Map(
        location=DEFAULT_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles="OpenStreetMap",
        control_scale=True,
    )

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

    return AreaBounds(
        south=min(lats),
        west=min(lons),
        north=max(lats),
        east=max(lons),
    )


def show_result(result) -> None:
    network = result.grid_network
    summary = result.summary_dict()

    st.subheader("Output für grid_model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buses / Knoten", result.bus_count)
    c2.metric("Lines", result.line_count)
    c3.metric("Households", result.household_count)
    c4.metric("Transformers", result.transformer_count)

    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)

    st.write("**Koordinaten und OSM-Zählwerte**")
    st.json(
        {
            "bbox": summary["bounds"],
            "selected_coordinate_count": summary["selected_coordinate_count"],
            "raw_osm_node_count": summary["raw_osm_node_count"],
            "raw_osm_way_count": summary["raw_osm_way_count"],
        }
    )

    grid_json = network.model_dump_json(indent=2)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)

    st.download_button(
        label="grid_network.json herunterladen",
        data=grid_json,
        file_name="grid_network.json",
        mime="application/json",
    )

    st.download_button(
        label="map_ui_summary.json herunterladen",
        data=summary_json,
        file_name="map_ui_summary.json",
        mime="application/json",
    )

    with st.expander("GridNetwork JSON anzeigen"):
        st.code(grid_json, language="json")


if __name__ == "__main__":
    main()
