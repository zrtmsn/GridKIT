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
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from grid_model.builder import OSMNetworkBuilder
from map_ui.osm_fetcher import AreaBounds


DEFAULT_CENTER = (49.0069, 8.4037)  # Karlsruhe
DEFAULT_ZOOM = 15


def main() -> None:
    st.set_page_config(page_title="GridKIT map_ui", layout="wide")

    st.title("GridKIT Karte")
    st.caption(
        "Kartenansicht → Bereich auswählen → GridNetwork erzeugen → Netzansicht visualisieren"
    )

    if "built_network" not in st.session_state:
        st.session_state["built_network"] = None
    if "built_bounds" not in st.session_state:
        st.session_state["built_bounds"] = None
    if "built_scenario" not in st.session_state:
        st.session_state["built_scenario"] = None

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
        fmap = make_base_map()
        map_data = st_folium(
            fmap,
            height=650,
            width=None,
            returned_objects=["last_active_drawing", "all_drawings"],
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

                except Exception as exc:
                    st.error("Das GridNetwork konnte nicht mit grid_model erzeugt werden.")
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


def show_network_visualization(network, selected_bounds: AreaBounds) -> None:
    st.subheader("Netzansicht / Netzwerkkonfiguration")

    st.markdown(
        """
        **Legende**
        - 🟠 Transformator-Bus
        - 🟢 Haushalts-Bus
        - 🔵 Sonstiger Bus
        - Türkise Linien = Leitungen
        """
    )

    fmap, missing_bus_coords, missing_line_coords = create_network_map(network, selected_bounds)

    st_folium(
        fmap,
        height=700,
        width=None,
        returned_objects=[],
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


def create_network_map(network, selected_bounds: AreaBounds) -> tuple[folium.Map, int, int]:
    buses_with_coords = [
        bus for bus in network.buses
        if bus.x_coord is not None and bus.y_coord is not None
    ]

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

    # ausgewählten Bereich anzeigen
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

    bus_lookup = {bus.bus_id: bus for bus in network.buses}
    household_ids = set(network.household_bus_ids)

    transformer_bus_ids = set()
    for trafo in network.transformers:
        transformer_bus_ids.add(trafo.hv_bus)
        transformer_bus_ids.add(trafo.lv_bus)

    line_group = folium.FeatureGroup(name="Leitungen", show=True)
    household_group = folium.FeatureGroup(name="Haushalte", show=True)
    transformer_group = folium.FeatureGroup(name="Transformator-Busse", show=True)
    bus_group = folium.FeatureGroup(name="Sonstige Busse", show=True)

    missing_line_coords = add_network_lines(line_group, network, bus_lookup)
    missing_bus_coords = add_network_buses(
        household_group=household_group,
        transformer_group=transformer_group,
        bus_group=bus_group,
        network=network,
        household_ids=household_ids,
        transformer_bus_ids=transformer_bus_ids,
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
        from_bus = bus_lookup.get(line.from_bus)
        to_bus = bus_lookup.get(line.to_bus)

        if from_bus is None or to_bus is None:
            missing_line_coords += 1
            continue

        if (
            from_bus.x_coord is None or from_bus.y_coord is None
            or to_bus.x_coord is None or to_bus.y_coord is None
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
) -> int:
    missing_bus_coords = 0

    for bus in network.buses:
        if bus.x_coord is None or bus.y_coord is None:
            missing_bus_coords += 1
            continue

        popup_html = make_bus_popup(
            bus=bus,
            household_ids=household_ids,
            transformer_bus_ids=transformer_bus_ids,
            network=network,
        )

        if bus.bus_id in transformer_bus_ids:
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

        elif bus.bus_id in household_ids:
            folium.CircleMarker(
                location=[bus.y_coord, bus.x_coord],
                radius=6,
                color="#238b45",
                fill=True,
                fill_color="#41ab5d",
                fill_opacity=0.9,
                weight=1,
                tooltip=f"Haushalt: {bus.bus_id}",
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
                tooltip=f"Bus: {bus.bus_id}",
                popup=folium.Popup(popup_html, max_width=350),
            ).add_to(bus_group)

    return missing_bus_coords


def make_bus_popup(bus, household_ids: set[str], transformer_bus_ids: set[str], network) -> str:
    roles: list[str] = []

    if bus.bus_id in transformer_bus_ids:
        roles.append("Transformator-Bus")
    if bus.bus_id in household_ids:
        roles.append("Haushalt")
    if not roles:
        roles.append("Standard-Bus")

    has_ev = bus.bus_id in getattr(network, "ev_availability", {})
    has_load_profile = bus.bus_id in getattr(network, "household_load_profile_kw", {})

    return f"""
    <b>Bus ID:</b> {bus.bus_id}<br>
    <b>Rolle:</b> {", ".join(roles)}<br>
    <b>Nominalspannung (kV):</b> {bus.v_nom_kv}<br>
    <b>x_coord:</b> {bus.x_coord}<br>
    <b>y_coord:</b> {bus.y_coord}<br>
    <b>Load Profile:</b> {"Ja" if has_load_profile else "Nein"}<br>
    <b>EV Availability:</b> {"Ja" if has_ev else "Nein"}
    """


if __name__ == "__main__":
    main()