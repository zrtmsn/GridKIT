#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
# dashboard_prototype/app.py
#
# Streamlit Dashboard Prototype für Overload-Visualisierung.
#
# Run: python -m streamlit run src/GridKIT/dashboard_prototype/app.py
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import folium
import streamlit as st
from streamlit_folium import st_folium

from GridKIT.dashboard_prototype.data_loader import (
    load_timelines,
    extract_overloaded_lines,
    get_line_coordinates,
    get_household_coordinates,
    get_transformer_coordinates,
)


DEFAULT_CENTER = (49.0069, 8.4037)  # Karlsruhe
DEFAULT_ZOOM = 15


def create_overload_map(
    network_path: str,
    timelines: list[dict],
    selected_scenario: str,
    current_step: int,
) -> folium.Map:
    """Erstelle Folium-Karte mit überlasteten Lines."""
    
    line_coords = get_line_coordinates(network_path)
    households = get_household_coordinates(network_path)
    transformers = get_transformer_coordinates(network_path)
    overload_data = extract_overloaded_lines(timelines)
    
    st.info(f"🔍 Geladen: {len(households)} Haushalte, {len(transformers)} Transformatoren aus '{network_path}'")
    
    if selected_scenario not in overload_data:
        selected_scenario = list(overload_data.keys())[0]
    
    scenario_data = overload_data[selected_scenario]
    entry = scenario_data['entry']
    
    overloaded_this_step = set()
    for step_idx, lines in scenario_data['overload_by_step'].items():
        if step_idx == current_step:
            overloaded_this_step = lines
            break
    
    # Auch Transformers prüfen!
    overloaded_transformers_this_step = set()
    for step_idx, trafos in enumerate(entry.get('overloaded_transformers', [])):
        if step_idx == current_step and trafos:
            overloaded_transformers_this_step.update(trafos)
    
    all_overloaded = scenario_data['all_lines']
    
    # Zentrum berechnen aus Haushalts-Koordinaten
    if households:
        center_lat = sum(h['coords'][0] for h in households) / len(households)
        center_lon = sum(h['coords'][1] for h in households) / len(households)
        
        # Prüfen ob Koordinaten meterisch sind (sehr kleine Werte < 1) → dann zu GPS konvertieren
        # Echte GPS-Koordinaten: lat ~49, lon ~8 (Karlsruhe)
        # Meterische Koordinaten: lat ~0-5, lon ~0-5 (stub_network)
        if abs(center_lat) < 1 and abs(center_lon) < 1:
            # Stub-Network hat meterische Koordinaten → nach Karlsruhe verschieben
            offset_lat = 49.0069  # Karlsruhe
            offset_lon = 8.4037
            scale = 0.0001  # ~10m pro Einheit
            
            center_lat = offset_lat + center_lat * scale
            center_lon = offset_lon + center_lon * scale
            zoom = 16
            use_gps_scale = True
        else:
            # Echte GPS-Koordinaten → unverändert lassen, näher ran zoomen
            zoom = 18  # Stärker reinzoomen für Service-Lines
            use_gps_scale = False
    else:
        center_lat, center_lon = 49.0069, 8.4037  # Karlsruhe Default
        zoom = 16
        use_gps_scale = False
    
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
    
    # Helper-Funktion zum Skalieren der Koordinaten
    def scale_coords(lat, lon):
        if use_gps_scale:
            return (49.0069 + lat * 0.0001, 8.4037 + lon * 0.0001)
        return (lat, lon)  # Echte GPS-Koordinaten unverändert
    
    line_group = folium.FeatureGroup(name="Leitungen (normal)")
    overload_group = folium.FeatureGroup(name="Überlastete Lines")
    household_group = folium.FeatureGroup(name="Haushalte")
    
    for line_id, coords in line_coords.items():
        if line_id in overloaded_this_step:
            continue
        
        from_scaled = scale_coords(coords['from'][0], coords['from'][1])
        to_scaled = scale_coords(coords['to'][0], coords['to'][1])
        
        folium.PolyLine(
            locations=[from_scaled, to_scaled],
            color="#00ced1",
            weight=2,
            opacity=0.8,
            tooltip=f"Line: {line_id}",
        ).add_to(line_group)
    
    for line_id in overloaded_this_step:
        if line_id not in line_coords:
            continue
        coords = line_coords[line_id]
        from_scaled = scale_coords(coords['from'][0], coords['from'][1])
        to_scaled = scale_coords(coords['to'][0], coords['to'][1])
        folium.PolyLine(
            locations=[from_scaled, to_scaled],
            color="#ff0000",
            weight=6,
            opacity=1.0,
            tooltip=f"⚠️ OVERLOAD: {line_id}",
        ).add_to(overload_group)
    
    for hh in households:
        scaled = scale_coords(hh['coords'][0], hh['coords'][1])
        folium.CircleMarker(
            location=scaled,
            radius=8,
            color="#238b45",
            fill=True,
            fill_color="#41ab5d",
            fill_opacity=1.0,
            weight=2,
            tooltip=f"Haushalt: {hh['bus_id']}",
        ).add_to(household_group)
    
    # Force household_group to be visible by default
    household_group.add_to(fmap)
    
    # Transformator als GRÜNE Punkte anzeigen (normal)
    trafo_group = folium.FeatureGroup(name="Transformatoren")
    for trafo in transformers:
        scaled = scale_coords(trafo['coords'][0], trafo['coords'][1])
        folium.CircleMarker(
            location=scaled,
            radius=12,
            color="#006400",
            fill=True,
            fill_color="#228B22",
            fill_opacity=1.0,
            weight=3,
            tooltip=f"⚡ Transformator: {trafo['trafo_id']}",
        ).add_to(trafo_group)
    
    trafo_group.add_to(fmap)
    
    # Überlastete Transformatoren ROT markieren!
    for trafo_id in overloaded_transformers_this_step:
        # Transformer-Koordinaten finden - Match auf LV-Bus ID!
        found = False
        for trafo in transformers:
            # Match auf lv_bus (wo Overload gemessen wird) statt trafo_id
            if trafo.get('lv_bus') == trafo_id:
                found = True
                scaled = scale_coords(trafo['coords'][0], trafo['coords'][1])
                folium.CircleMarker(
                    location=scaled,
                    radius=16,
                    color="#ff0000",
                    fill=True,
                    fill_color="#ff0000",
                    fill_opacity=1.0,
                    weight=4,
                    tooltip=f"🚨 OVERLOAD: {trafo_id}",
                ).add_to(overload_group)
        
        if not found:
            st.error(f"❌ Transformer '{trafo_id}' NICHT GEFUNDEN in transformers Liste!")
    
    # Add all groups to map explicitly
    line_group.add_to(fmap)
    overload_group.add_to(fmap)
    
    folium.LayerControl(collapsed=False).add_to(fmap)
    
    return fmap, entry, len(overloaded_this_step), len(all_overloaded), len(overloaded_transformers_this_step)


def main() -> None:
    st.set_page_config(page_title="GridKIT Overload Dashboard", layout="wide")
    
    st.title("⚡ GridKIT Overload Dashboard")
    st.caption("Prototype: Visualisierung von Netzüberlastungen aus timelines.json")
    
    with st.sidebar:
        st.header("Konfiguration")
        
        results_dir = st.text_input(
            "Output Verzeichnis",
            value="outputs_karlsruhe_very_weak_v3",
            help="Verzeichnis mit timelines.json"
        )
        
        network_path = st.text_input(
            "Netzwerk-Pfad",
            value="data/17_households_very_weak.json",
            help="Pfad zur Netzwerk-JSON"
        )
        
        st.divider()
        
        if st.button("Daten laden", type="primary"):
            try:
                timelines = load_timelines(results_dir)
                st.session_state['timelines'] = timelines
                st.success(f"✅ {len(timelines)} Szenarien geladen!")
            except FileNotFoundError as e:
                st.error(str(e))
                st.session_state.pop('timelines', None)
    
    if 'timelines' not in st.session_state:
        st.info("👈 Klicke auf 'Daten laden' in der Sidebar um zu starten!")
        return
    
    timelines = st.session_state['timelines']
    
    # Tabs: Overload Map + Metrics + Training
    tab_map, tab_metrics, tab_training = st.tabs(["🗺️ Overload Map", "📊 Metrics (PNGs)", "🎯 Training (Rewards)"])
    
    with tab_map:
        render_overload_map_tab(network_path, timelines)
    
    with tab_metrics:
        render_metrics_tab(results_dir)
    
    with tab_training:
        render_training_tab(results_dir)


def render_overload_map_tab(network_path: str, timelines: list[dict]) -> None:
    """Rendert den Overload Map Tab."""
    
    scenario_names = [
        f"{e['scenario']} @ {int(e['penetration'] * 100)}%"
        for e in timelines
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_scenario = st.selectbox(
            "Szenario auswählen",
            options=scenario_names,
            index=0,
            key="map_scenario"
        )
    
    with col2:
        current_step = st.slider(
            "Zeitstep (0-95)",
            min_value=0,
            max_value=95,
            value=0,
            key="map_step"
        )
    
    try:
        fmap, entry, num_overloads_now, num_overloads_total, num_trafo_overloads = create_overload_map(
            network_path=network_path,
            timelines=timelines,
            selected_scenario=selected_scenario,
            current_step=current_step,
        )
        
        st_folium(fmap, height=600, width=None)
        
        st.divider()
        
        col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
        
        with col_stat1:
            st.metric("Overloaded Lines (dieser Step)", value=num_overloads_now)
        
        with col_stat2:
            st.metric("Overloaded Lines (gesamt)", value=num_overloads_total)
        
        with col_stat3:
            st.metric("Overloaded Transformers (dieser Step)", value=num_trafo_overloads)
        
        with col_stat4:
            curtail_steps = sum(1 for x in entry.get('curtailment', []) if x)
            st.metric("Curtailment Steps", value=curtail_steps)
        
        with col_stat5:
            st.metric("Penetration", value=f"{int(entry['penetration'] * 100)}%")
        
        # Überlastungen anzeigen
        has_overloads = num_overloads_now > 0 or num_trafo_overloads > 0
        
        if has_overloads:
            st.warning(f"⚠️ **Step {current_step}**: {num_overloads_now} Line(s) + {num_trafo_overloads} Transformer(s) überlastet!")
            
            overload_data = extract_overloaded_lines(timelines)
            scenario_data = overload_data[selected_scenario]
            
            # Line-Overloads anzeigen
            if num_overloads_now > 0:
                st.write("**Betroffene Lines:**")
                for step_idx, lines in scenario_data['overload_by_step'].items():
                    if step_idx == current_step:
                        for line_id in sorted(lines):
                            st.write(f"- `{line_id}`")
                        break
            
            # Transformer-Overloads anzeigen
            if num_trafo_overloads > 0:
                st.write("**Betroffene Transformatoren:**")
                for step_idx, trafos in enumerate(entry.get('overloaded_transformers', [])):
                    if step_idx == current_step and trafos:
                        for trafo_id in sorted(trafos):
                            st.write(f"- `{trafo_id}`")
                        break
        else:
            st.success(f"✅ **Step {current_step}**: Keine Überlastungen")
        
        with st.expander("ℹ️ Hinweise zur Visualisierung"):
            st.write("""
            - **Türkise Lines**: Normale Leitungen (< 1.0 p.u.)
            - **Rote Lines**: Überlastete Leitungen (> 1.0 p.u.) im aktuellen Step
            - **Grüne Punkte**: Haushalte
            - Slider: 96 Steps × 15min = 24h
            """)
        
    except Exception as e:
        st.error(f"Fehler beim Erstellen der Karte: {e}")
        st.exception(e)


def render_metrics_tab(results_dir: str) -> None:
    """Rendert den Metrics Tab mit PNGs aus plot_results."""
    import os
    from pathlib import Path
    
    graphs_dir = Path(results_dir) / "graphs"
    
    st.header("📊 Metriken aus `plot_results`")
    st.write(f"Verzeichnis: `{graphs_dir}`")
    
    if not graphs_dir.exists():
        st.warning(f"⚠️ Kein `graphs` Verzeichnis gefunden unter `{graphs_dir}`")
        st.info("""
        **Erstelle PNGs mit:**
        ```bash
        export PYTHONPATH=src/GridKIT:src
        python -m GridKIT.scripts.plot_results --results outputs_karlsruhe_very_weak_v3 --out outputs_karlsruhe_very_weak_v3/graphs
        ```
        """)
        return
    
    png_files = sorted(list(graphs_dir.glob("*.png")))
    
    if not png_files:
        st.warning("Keine PNG-Dateien gefunden!")
        return
    
    st.success(f"✅ {len(png_files)} Diagramme gefunden!")
    
    # Zeige PNGs in einem Grid
    for png_path in png_files:
        st.subheader(png_path.stem.replace("_", " ").title())
        st.image(str(png_path), use_container_width=True)
        st.divider()


def render_training_tab(results_dir: str) -> None:
    """
    Rendert den Training Tab mit Reward-Konvergenz.
    
    ═══════════════════════════════════════════════════════════════
    DATENBASIS
    ═══════════════════════════════════════════════════════════════
    
    Quelle: `checkpoints/pen_XX/iteration_metrics.json`
    
    Diese Datei wird während des Trainings von `run_experiment.py` 
    erstellt und enthält Rohdaten von RLlib für jede Trainings-Iteration.
    
    Pro Trainings-Iteration (default: 40, hier: 3) werden gespeichert:
    - Episode Returns (mean/min/max) über alle Episoden dieser Iteration
    - Policy-Metriken (Loss, Entropy, KL-Divergenz) pro Policy
    - Value Function Metriken (Loss, explained variance)
    - Timing-Informationen
    
    WICHTIG: Jede Iteration = Ein `algo.train()` Call bei RLlib.
    Eine Iteration sammelt Daten von mehreren Episoden (parallel Workers).
    Die angezeigten Werte sind Durchschnitte über alle Episoden der 
    jeweiligen Iteration.
    
    Dateipfad: outputs/{results_dir}/checkpoints/pen_{20,40,60}/iteration_metrics.json
    ═══════════════════════════════════════════════════════════════
    """
    import json
    from pathlib import Path
    
    st.header("🎯 Training Metriken")
    st.markdown("""
    **Datenbasis:** Pro Trainings-Iteration aggregierte RLlib-Metriken.
    
    Jede Iteration sammelt Daten von mehreren parallelen Episoden. 
    Die Charts zeigen Durchschnitte über alle Episoden einer Iteration.
    """)
    
    checkpoints_dir = Path(results_dir) / "checkpoints"
    
    if not checkpoints_dir.exists():
        st.warning(f"⚠️ Kein Checkpoints-Verzeichnis gefunden!")
        return
    
    metrics_files = sorted(list(checkpoints_dir.glob("pen_*/iteration_metrics.json")))
    
    if not metrics_files:
        st.warning("Keine `iteration_metrics.json` Dateien gefunden!")
        return
    
    st.success(f"✅ {len(metrics_files)} Penetration-Level gefunden!")
    
    _render_training_charts(checkpoints_dir, metrics_files)


def _render_training_charts(checkpoints_dir: Path, metrics_files: list) -> None:
    """Helper für Training Charts."""
    import json
    
    try:
        import plotly.graph_objects as go
        import pandas as pd
        
        # Parent directory name verwenden (pen_20, pen_40, etc.)
        penetration_options = [f.parent.name.replace("pen_", "") for f in metrics_files]
        selected_pen = st.selectbox("Penetration-Level", options=penetration_options, index=len(penetration_options)-1)
        
        # Korrekter Pfad zur iteration_metrics.json
        metrics_file = checkpoints_dir / f"pen_{selected_pen}" / "iteration_metrics.json"
        with open(metrics_file) as f:
            raw_data = json.load(f)
        
        iterations, rewards_mean, rewards_min, rewards_max = [], [], [], []
        policy_loss, entropy, vf_loss = [], [], []
        
        for entry in raw_data:
            iterations.append(entry.get('training_iteration', len(iterations) + 1))
            env_runners = entry.get('env_runners', {})
            rewards_mean.append(env_runners.get('episode_return_mean', 0.0))
            rewards_min.append(env_runners.get('episode_return_min', 0.0))
            rewards_max.append(env_runners.get('episode_return_max', 0.0))
            
            # RLlib speichert unter 'learners' mit per-policy keys
            learners = entry.get('learners', {})
            ev_policy = learners.get('ev_policy', {})
            policy_loss.append(ev_policy.get('policy_loss', 0.0))
            entropy.append(ev_policy.get('entropy', 0.0))
            vf_loss.append(ev_policy.get('vf_loss', 0.0))
        
        # Reward Konvergenz
        st.subheader("💰 Reward Konvergenz")
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(x=iterations, y=rewards_mean, mode='lines+markers', name='Mean'))
        fig_reward.add_trace(go.Scatter(x=iterations, y=rewards_min, mode='lines', name='Min', line=dict(dash='dot')))
        fig_reward.add_trace(go.Scatter(x=iterations, y=rewards_max, mode='lines', name='Max', line=dict(dash='dot')))
        fig_reward.update_layout(title="Episode Rewards", xaxis_title="Iteration", yaxis_title="Reward", height=400)
        st.plotly_chart(fig_reward, use_container_width=True)
        
        # Policy Metrics
        st.subheader("🧠 Policy Qualität")
        col1, col2 = st.columns(2)
        with col1:
            fig_pl = go.Figure()
            fig_pl.add_trace(go.Scatter(x=iterations, y=policy_loss, mode='lines+markers', name='Policy Loss'))
            fig_pl.update_layout(title="Policy Loss", height=300)
            st.plotly_chart(fig_pl, use_container_width=True)
        with col2:
            fig_ent = go.Figure()
            fig_ent.add_trace(go.Scatter(x=iterations, y=entropy, mode='lines+markers', name='Entropy', marker_color='orange'))
            fig_ent.update_layout(title="Entropy", height=300)
            st.plotly_chart(fig_ent, use_container_width=True)
        
        # VF Loss
        st.subheader("📉 Value Function")
        fig_vf = go.Figure()
        fig_vf.add_trace(go.Scatter(x=iterations, y=vf_loss, mode='lines+markers', name='VF Loss', marker_color='green'))
        fig_vf.update_layout(title="Value Function Loss", height=300)
        st.plotly_chart(fig_vf, use_container_width=True)
        
    except ImportError:
        st.error("Plotly/Pandas nicht installiert! `pip install plotly pandas`")


if __name__ == "__main__":
    main()
