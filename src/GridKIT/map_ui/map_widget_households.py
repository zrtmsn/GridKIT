# ─────────────────────────────────────────────────────────────
# map_ui/map_widget_households.py
#
# Household configuration UI for map_widget.py.
#
# This file contains:
# - GridCreator household device summary
# - global target sliders
# - individual household overrides
# - household_configuration.json creation
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

import streamlit as st

from map_ui.household_config import (
    bool_from_choice,
    build_household_configuration,
    choice_index_from_bool,
    default_scenario_assumptions,
    gridcreator_defaults,
    json_dumps_pretty,
    select_households_by_share,
)
from map_ui.osm_fetcher import AreaBounds
from map_ui.transformer_network_filter import device_summary


LOAD_PROFILE_MODE_ORIGINAL = "Originales Lastprofil unverändert verwenden"
LOAD_PROFILE_MODE_GLOBAL = "Globale Szenario-Skalierung anwenden"
LOAD_PROFILE_MODE_CUSTOM = "Eigene Skalierung für diesen Haushalt festlegen"


def show_household_configuration(network, selected_bounds: AreaBounds) -> dict[str, Any] | None:
    st.subheader("Haushaltskonfiguration")

    selected_trafo_id = st.session_state.get("selected_trafo_id")
    household_ids = sorted(str(bus_id) for bus_id in getattr(network, "household_bus_ids", []))

    if selected_trafo_id is None:
        st.info(
            "Es ist aktuell kein Trafo-Filter aktiv. "
            "Die Haushaltskonfiguration bezieht sich auf alle Haushalts-/Last-Busse "
            "im vollständig erzeugten GridNetwork des ausgewählten Kartenbereichs."
        )
    else:
        st.info(
            f"Die Haushaltskonfiguration bezieht sich aktuell ausschließlich auf "
            f"das ausgewählte Trafo-Netz von Transformator **{selected_trafo_id}**. "
            f"Es werden nur Haushalts-/Last-Busse angezeigt und konfiguriert, "
            f"die in diesem gefilterten Teilnetz enthalten sind."
        )

    if not household_ids:
        if selected_trafo_id is None:
            st.warning(
                "Im erzeugten GridNetwork wurden keine Haushalts-/Last-Busse gefunden. "
                "Die Haushaltskonfiguration kann deshalb nicht angewendet werden."
            )
        else:
            st.warning(
                "Im ausgewählten Trafo-Netz wurden keine Haushalts-/Last-Busse gefunden. "
                "Die Haushaltskonfiguration kann deshalb nicht angewendet werden."
            )
        return None

    if st.session_state.pop("scenario_assumptions_saved_message", False):
        st.success("Konfiguration wurde gespeichert.")

    if st.session_state.pop("scenario_assumptions_reset_message", False):
        st.success("Konfiguration wurde auf die GridCreator-Basisdaten zurückgesetzt.")

    saved_household = st.session_state.pop("household_saved_message", None)
    if saved_household:
        st.success(f"Individuelle Anpassung für {saved_household} gespeichert.")

    reset_household = st.session_state.pop("household_reset_message", None)
    if reset_household:
        st.info(f"Individuelle Anpassung für {reset_household} wurde gelöscht.")

    st.session_state.setdefault("household_overrides", {})
    st.session_state.setdefault("scenario_assumptions", default_scenario_assumptions())
    st.session_state.setdefault("global_device_targets", None)
    st.session_state.setdefault("global_device_target_scope", None)
    st.session_state.setdefault("scenario_config_version", 0)
    st.session_state.setdefault("household_config_version", 0)

    individual_overrides: dict[str, dict[str, Any]] = st.session_state["household_overrides"]
    saved_assumptions: dict[str, Any] = st.session_state["scenario_assumptions"]

    gridcreator_devices = gridcreator_defaults(network)
    uses_gridcreator_defaults = bool(gridcreator_devices)

    configuration_scope = (
        f"{getattr(network, 'network_id', 'unknown_network')}|"
        f"{selected_trafo_id or 'complete_network'}|"
        f"{len(household_ids)}"
    )

    if st.session_state.get("global_device_target_scope") != configuration_scope:
        st.session_state["global_device_targets"] = None
        st.session_state["global_device_target_scope"] = configuration_scope
        st.session_state["scenario_config_version"] += 1

    gridcreator_summary = device_summary(network)

    baseline_targets = baseline_device_targets(
        network=network,
        saved_assumptions=saved_assumptions,
        uses_gridcreator_defaults=uses_gridcreator_defaults,
    )

    saved_global_targets = st.session_state.get("global_device_targets")
    active_targets = saved_global_targets or baseline_targets
    global_targets_active = saved_global_targets is not None

    st.markdown("### GridCreator-Basisdaten und globale Anpassungen")

    if uses_gridcreator_defaults:
        st.info(
            "Für dieses Netz liegen echte GridCreator-Gerätedaten vor. "
            "Diese Daten werden als Ausgangszustand der Haushaltskonfiguration verwendet. "
            "Die unten gezeigten Werte beschreiben zuerst die erkannte Ausstattung im "
            "aktuellen Netz bzw. Trafo-Netz. Über die globalen Zielwerte und die "
            "individuellen Haushaltseinstellungen können diese Basisdaten anschließend "
            "für Szenarien angepasst werden."
        )
    else:
        st.info(
            "Für dieses Netz liegen keine vollständigen GridCreator-Gerätedaten pro Haushalt vor. "
            "Die Ausstattung wird deshalb über globale Zielwerte verteilt. "
            "Sobald echte GridCreator-Gerätedaten vorhanden sind, werden diese automatisch "
            "als Ausgangszustand verwendet."
        )

    show_gridcreator_device_metrics(gridcreator_summary)

    st.caption(
        "GridCreator/grid_model liefert für jeden Haushalt ein originales Lastprofil. "
        "Dieses Lastprofil beschreibt den zeitlichen Stromverbrauch, zum Beispiel über "
        "96 Zeitschritte eines Tages. Die Skalierung ist keine zusätzliche Originalinformation "
        "aus GridCreator, sondern eine optionale Szenario-Annahme der Map UI. "
        "Eine Skalierung von 1.00 bedeutet: originales Lastprofil unverändert."
    )

    widget_suffix = st.session_state["scenario_config_version"]

    high_col1, high_col2, high_col3, high_col4 = st.columns(4)

    draft_ev_share_percent = high_col1.slider(
        "Ziel-Anteil Haushalte mit EV (%)",
        min_value=0.0,
        max_value=100.0,
        value=percent_slider_value(active_targets["ev_share_percent"]),
        step=0.1,
        format="%.1f",
        help=(
            "Ausgangswert ist der aus GridCreator erkannte EV-Anteil. "
            "Nach dem Speichern kann dieser Zielwert für das Szenario geändert werden."
        ),
        key=f"draft_ev_share_percent_{widget_suffix}",
    )

    draft_heat_pump_share_percent = high_col2.slider(
        "Ziel-Anteil Haushalte mit WP (%)",
        min_value=0.0,
        max_value=100.0,
        value=percent_slider_value(active_targets["heat_pump_share_percent"]),
        step=0.1,
        format="%.1f",
        help=(
            "Ausgangswert ist der aus GridCreator erkannte Wärmepumpen-Anteil. "
            "Nach dem Speichern kann dieser Zielwert für das Szenario geändert werden."
        ),
        key=f"draft_heat_pump_share_percent_{widget_suffix}",
    )

    draft_battery_share_percent = high_col3.slider(
        "Ziel-Anteil Haushalte mit Batterie (%)",
        min_value=0.0,
        max_value=100.0,
        value=percent_slider_value(active_targets["battery_share_percent"]),
        step=0.1,
        format="%.1f",
        help=(
            "Ausgangswert ist der aus GridCreator erkannte Batterie-Anteil. "
            "Nach dem Speichern kann dieser Zielwert für das Szenario geändert werden."
        ),
        key=f"draft_battery_share_percent_{widget_suffix}",
    )

    draft_pv_share_percent = high_col4.slider(
        "Ziel-Anteil Haushalte mit PV (%)",
        min_value=0.0,
        max_value=100.0,
        value=percent_slider_value(active_targets["pv_share_percent"]),
        step=0.1,
        format="%.1f",
        help=(
            "Ausgangswert ist der aus GridCreator erkannte PV-Anteil. "
            "Nach dem Speichern kann dieser Zielwert für das Szenario geändert werden."
        ),
        key=f"draft_pv_share_percent_{widget_suffix}",
    )

    draft_global_load_scaling_factor = st.number_input(
        "Globale Lastprofil-Skalierung für Szenarien",
        min_value=0.1,
        max_value=5.0,
        value=float(saved_assumptions.get("global_load_scaling_factor", 1.0)),
        step=0.1,
        format="%.2f",
        help=(
            "Diese Skalierung wird nur als Szenario-Annahme verwendet. "
            "1.00 bedeutet: originale Lastprofile unverändert. "
            "1.20 bedeutet: alle aktuell angezeigten Lastprofile werden um 20 % erhöht. "
            "0.80 bedeutet: alle aktuell angezeigten Lastprofile werden um 20 % reduziert."
        ),
        key=f"draft_global_load_scaling_factor_{widget_suffix}",
    )

    with st.expander("Erweiterte Einstellungen"):
        draft_selection_seed = st.number_input(
            "Zufallswert für automatische Zielwert-Verteilung",
            min_value=0,
            max_value=99999,
            value=int(saved_assumptions.get("selection_seed", 42)),
            step=1,
            help=(
                "Dieser Wert ist nur für die automatische Verteilung globaler Zielanteile relevant. "
                "Beispiel: Wenn 30 % der Haushalte EV haben sollen, entscheidet dieser Wert, "
                "welche konkreten Haushalte ausgewählt werden. Gleicher Wert bedeutet: "
                "gleiche Verteilung."
            ),
            key=f"draft_selection_seed_{widget_suffix}",
        )

    draft_device_targets = {
        "ev_share_percent": float(draft_ev_share_percent),
        "heat_pump_share_percent": float(draft_heat_pump_share_percent),
        "battery_share_percent": float(draft_battery_share_percent),
        "pv_share_percent": float(draft_pv_share_percent),
    }

    draft_assumptions = {
        **saved_assumptions,
        "ev_share_percent": float(draft_ev_share_percent),
        "heat_pump_share_percent": float(draft_heat_pump_share_percent),
        "battery_share_percent": float(draft_battery_share_percent),
        "pv_share_percent": float(draft_pv_share_percent),
        "global_load_scaling_factor": float(draft_global_load_scaling_factor),
        "selection_seed": int(draft_selection_seed),
    }

    action_col1, action_col2 = st.columns(2)

    if action_col1.button("Globale Konfiguration speichern"):
        st.session_state["global_device_targets"] = draft_device_targets
        st.session_state["scenario_assumptions"] = draft_assumptions
        st.session_state["scenario_config_version"] += 1
        st.session_state["scenario_assumptions_saved_message"] = True
        st.rerun()

    if action_col2.button("Auf GridCreator-Basisdaten zurücksetzen"):
        reset_assumptions = default_scenario_assumptions()
        reset_assumptions["global_load_scaling_factor"] = 1.0
        reset_assumptions["selection_seed"] = int(saved_assumptions.get("selection_seed", 42))

        st.session_state["global_device_targets"] = None
        st.session_state["scenario_assumptions"] = reset_assumptions
        st.session_state["scenario_config_version"] += 1
        st.session_state["scenario_assumptions_reset_message"] = True
        st.rerun()

    if global_targets_active:
        st.warning(
            "Es sind gespeicherte globale Zielwerte aktiv. "
            "Die aktuelle Haushaltskonfiguration basiert deshalb nicht mehr ausschließlich "
            "auf den ursprünglichen GridCreator-Gerätedaten."
        )

    if (
        draft_device_targets != active_targets
        or float(draft_global_load_scaling_factor) != float(saved_assumptions.get("global_load_scaling_factor", 1.0))
        or int(draft_selection_seed) != int(saved_assumptions.get("selection_seed", 42))
    ):
        st.warning(
            "Es gibt nicht gespeicherte Änderungen in der globalen Konfiguration. "
            "Klicke auf „Globale Konfiguration speichern“, damit diese Werte übernommen werden."
        )

    saved_global_load_scaling_factor = float(saved_assumptions.get("global_load_scaling_factor", 1.0))
    saved_selection_seed = int(saved_assumptions.get("selection_seed", 42))

    effective_overrides = build_effective_household_overrides(
        household_ids=household_ids,
        device_targets=active_targets,
        selection_seed=saved_selection_seed,
        individual_overrides=individual_overrides,
        generate_global_overrides=global_targets_active or not uses_gridcreator_defaults,
    )

    st.markdown("### Individuelle Anpassung einzelner Haushalte")

    selected_household = st.selectbox(
        "Haushalt / Lastpunkt auswählen",
        household_ids,
        help="Hier kann ein konkreter vorhandener Haushalts-/Last-Bus individuell angepasst werden.",
    )

    current_override = individual_overrides.get(selected_household, {})
    household_widget_suffix = st.session_state["household_config_version"]

    original_devices = gridcreator_devices.get(selected_household)
    automatic_values = automatic_device_values_for_household(
        household_id=selected_household,
        household_ids=household_ids,
        original_devices=original_devices,
        uses_gridcreator_defaults=uses_gridcreator_defaults,
        active_targets=active_targets,
        selection_seed=saved_selection_seed,
        global_targets_active=global_targets_active,
    )

    show_household_gridcreator_details(
        network=network,
        household_id=selected_household,
        original_devices=original_devices,
    )

    original_load_scaling_factor = grid_model_default_load_scaling_factor(
        network=network,
        household_id=selected_household,
    )

    low_col1, low_col2, low_col3, low_col4, low_col5 = st.columns(5)

    ev_choice = low_col1.selectbox(
        "EV für diesen Haushalt",
        options=[auto_label(automatic_values["has_ev"]), "Ja", "Nein"],
        index=choice_index_from_bool(current_override.get("has_ev")),
        key=f"ev_choice_{selected_household}_{household_widget_suffix}",
        help=auto_help(
            "EV",
            uses_gridcreator_defaults=uses_gridcreator_defaults,
            global_targets_active=global_targets_active,
        ),
    )

    heat_pump_choice = low_col2.selectbox(
        "WP für diesen Haushalt",
        options=[auto_label(automatic_values["has_heat_pump"]), "Ja", "Nein"],
        index=choice_index_from_bool(current_override.get("has_heat_pump")),
        key=f"heat_pump_choice_{selected_household}_{household_widget_suffix}",
        help=auto_help(
            "Wärmepumpe",
            uses_gridcreator_defaults=uses_gridcreator_defaults,
            global_targets_active=global_targets_active,
        ),
    )

    battery_choice = low_col3.selectbox(
        "Batterie für diesen Haushalt",
        options=[auto_label(automatic_values["has_battery"]), "Ja", "Nein"],
        index=choice_index_from_bool(current_override.get("has_battery")),
        key=f"battery_choice_{selected_household}_{household_widget_suffix}",
        help=auto_help(
            "Batterie",
            uses_gridcreator_defaults=uses_gridcreator_defaults,
            global_targets_active=global_targets_active,
        ),
    )

    pv_choice = low_col4.selectbox(
        "PV für diesen Haushalt",
        options=[auto_label(automatic_values["has_pv"]), "Ja", "Nein"],
        index=choice_index_from_bool(current_override.get("has_pv")),
        key=f"pv_choice_{selected_household}_{household_widget_suffix}",
        help=auto_help(
            "PV",
            uses_gridcreator_defaults=uses_gridcreator_defaults,
            global_targets_active=global_targets_active,
        ),
    )

    load_profile_mode_options = [
        LOAD_PROFILE_MODE_ORIGINAL,
        LOAD_PROFILE_MODE_GLOBAL,
        LOAD_PROFILE_MODE_CUSTOM,
    ]

    current_load_scaling_override = current_override.get("load_scaling_factor")

    if current_load_scaling_override is None:
        load_profile_mode_index = 0
    else:
        current_load_scaling_override = float(current_load_scaling_override)

        if abs(current_load_scaling_override - original_load_scaling_factor) < 1e-9:
            load_profile_mode_index = 0
        elif abs(current_load_scaling_override - saved_global_load_scaling_factor) < 1e-9:
            load_profile_mode_index = 1
        else:
            load_profile_mode_index = 2

    load_profile_mode = low_col5.selectbox(
        "Lastprofil für diesen Haushalt",
        options=load_profile_mode_options,
        index=load_profile_mode_index,
        key=f"load_profile_mode_{selected_household}_{household_widget_suffix}",
        help=(
            "Originales Lastprofil bedeutet: Die Zeitreihe aus GridCreator/grid_model "
            "wird unverändert verwendet. Die globale Szenario-Skalierung wendet den "
            f"aktuellen globalen Faktor {saved_global_load_scaling_factor:.2f} an. "
            "Eine eigene Skalierung speichert einen individuellen Faktor nur für diesen Haushalt."
        ),
    )

    configured_load_scaling_factor = low_col5.number_input(
        "Eigene Skalierung",
        min_value=0.1,
        max_value=5.0,
        value=float(
            current_override.get(
                "load_scaling_factor",
                original_load_scaling_factor,
            )
        ),
        step=0.1,
        format="%.2f",
        key=f"load_scaling_factor_{selected_household}_{household_widget_suffix}",
        disabled=load_profile_mode != LOAD_PROFILE_MODE_CUSTOM,
        help=(
            "Dieser Wert wird nur verwendet, wenn „Eigene Skalierung für diesen Haushalt festlegen“ "
            "ausgewählt ist. 1.00 bedeutet: Original-Lastprofil unverändert. "
            "1.20 bedeutet: Lastprofil dieses Haushalts um 20 % erhöhen."
        ),
    )

    low_col5.caption(
        f"Original: **unverändert** · "
        f"Globale Szenario-Skalierung: **{saved_global_load_scaling_factor:.2f}**"
    )

    individual_action_col1, individual_action_col2 = st.columns(2)

    if individual_action_col1.button(
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

        if load_profile_mode == LOAD_PROFILE_MODE_ORIGINAL:
            if abs(original_load_scaling_factor - saved_global_load_scaling_factor) > 1e-9:
                new_override["load_scaling_factor"] = float(original_load_scaling_factor)

        elif load_profile_mode == LOAD_PROFILE_MODE_CUSTOM:
            new_override["load_scaling_factor"] = float(configured_load_scaling_factor)

        if new_override:
            individual_overrides[selected_household] = new_override
            st.session_state["household_config_version"] += 1
            st.session_state["household_saved_message"] = selected_household
            st.rerun()
        else:
            individual_overrides.pop(selected_household, None)
            st.session_state["household_config_version"] += 1
            st.session_state["household_reset_message"] = selected_household
            st.rerun()

    if individual_action_col2.button(
        "Individuelle Anpassung löschen",
        key=f"delete_override_{selected_household}",
    ):
        individual_overrides.pop(selected_household, None)
        st.session_state["household_config_version"] += 1
        st.session_state["household_reset_message"] = selected_household
        st.rerun()

    household_configuration = build_household_configuration(
        network=network,
        selected_bounds=selected_bounds,
        household_ids=household_ids,
        ev_share_percent=float(active_targets["ev_share_percent"]),
        heat_pump_share_percent=float(active_targets["heat_pump_share_percent"]),
        global_load_scaling_factor=saved_global_load_scaling_factor,
        selection_seed=saved_selection_seed,
        household_overrides=effective_overrides,
    )

    st.markdown("### Zusammenfassung der aktuellen Haushaltskonfiguration")

    resolved = household_configuration["resolved"]

    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5, summary_col6 = st.columns(6)
    summary_col1.metric("Haushalte gesamt", len(household_ids))
    summary_col2.metric("Haushalte mit EV", len(resolved["ev_bus_ids"]))
    summary_col3.metric("Haushalte mit WP", len(resolved["heat_pump_bus_ids"]))
    summary_col4.metric("Haushalte mit Batterie", len(resolved["battery_bus_ids"]))
    summary_col5.metric("Haushalte mit PV", len(resolved["pv_bus_ids"]))
    summary_col6.metric("Individuelle Anpassungen", len(individual_overrides))

    with st.expander("household_configuration.json anzeigen"):
        st.json(household_configuration)

    st.download_button(
        label="household_configuration.json herunterladen",
        data=json_dumps_pretty(household_configuration),
        file_name="household_configuration.json",
        mime="application/json",
    )

    return household_configuration


def baseline_device_targets(
    network,
    saved_assumptions: dict[str, Any],
    uses_gridcreator_defaults: bool,
) -> dict[str, float]:
    """Return the default global target shares for the currently displayed network."""
    if uses_gridcreator_defaults:
        summary = device_summary(network)

        return {
            "ev_share_percent": float(summary["ev_share_percent"]),
            "heat_pump_share_percent": float(summary["heat_pump_share_percent"]),
            "battery_share_percent": float(summary["battery_share_percent"]),
            "pv_share_percent": float(summary["pv_share_percent"]),
        }

    return {
        "ev_share_percent": float(saved_assumptions.get("ev_share_percent", 0.0)),
        "heat_pump_share_percent": float(saved_assumptions.get("heat_pump_share_percent", 0.0)),
        "battery_share_percent": float(saved_assumptions.get("battery_share_percent", 0.0)),
        "pv_share_percent": float(saved_assumptions.get("pv_share_percent", 0.0)),
    }


def percent_slider_value(value: Any) -> float:
    """Return a bounded percentage value that Streamlit sliders can display."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0

    return max(0.0, min(100.0, round(numeric, 1)))


def show_gridcreator_device_metrics(summary: dict[str, Any]) -> None:
    """Show the real GridCreator device summary for the currently displayed network."""
    st.markdown("#### Erkannte GridCreator-Ausstattung")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Haushalte", summary["household_count"])
    c2.metric("EV", f"{summary['ev_count']} ({summary['ev_share_percent']} %)")
    c3.metric("WP", f"{summary['heat_pump_count']} ({summary['heat_pump_share_percent']} %)")
    c4.metric("Batterie", f"{summary['battery_count']} ({summary['battery_share_percent']} %)")
    c5.metric("PV", f"{summary['pv_count']} ({summary['pv_share_percent']} %)")


def build_effective_household_overrides(
    household_ids: list[str],
    device_targets: dict[str, float],
    selection_seed: int,
    individual_overrides: dict[str, dict[str, Any]],
    generate_global_overrides: bool,
) -> dict[str, dict[str, Any]]:
    """Combine global target shares with individual household overrides.

    Individual overrides have the highest priority. Global overrides are generated
    only when the user has saved global target values or when no real GridCreator
    device defaults exist.
    """
    if not generate_global_overrides:
        return individual_overrides

    ev_ids = select_households_by_share(
        household_ids,
        float(device_targets["ev_share_percent"]),
        seed=selection_seed,
        salt="ev",
    )
    heat_pump_ids = select_households_by_share(
        household_ids,
        float(device_targets["heat_pump_share_percent"]),
        seed=selection_seed,
        salt="heat_pump",
    )
    battery_ids = select_households_by_share(
        household_ids,
        float(device_targets["battery_share_percent"]),
        seed=selection_seed,
        salt="battery",
    )
    pv_ids = select_households_by_share(
        household_ids,
        float(device_targets["pv_share_percent"]),
        seed=selection_seed,
        salt="pv",
    )

    effective_overrides: dict[str, dict[str, Any]] = {}

    for household_id in household_ids:
        effective_overrides[household_id] = {
            "has_ev": household_id in ev_ids,
            "has_heat_pump": household_id in heat_pump_ids,
            "has_battery": household_id in battery_ids,
            "has_pv": household_id in pv_ids,
        }

    for household_id, override in individual_overrides.items():
        effective_overrides[household_id] = {
            **effective_overrides.get(household_id, {}),
            **override,
        }

    return effective_overrides


def automatic_device_values_for_household(
    household_id: str,
    household_ids: list[str],
    original_devices,
    uses_gridcreator_defaults: bool,
    active_targets: dict[str, float],
    selection_seed: int,
    global_targets_active: bool,
) -> dict[str, bool]:
    """Return the values used when the individual UI option remains automatic."""
    if uses_gridcreator_defaults and not global_targets_active:
        return {
            "has_ev": bool(read_object_value(original_devices, "ev", False)),
            "has_heat_pump": bool(read_object_value(original_devices, "heat_pump", False)),
            "has_battery": bool(read_object_value(original_devices, "battery", False)),
            "has_pv": bool(read_object_value(original_devices, "pv", False)),
        }

    ev_ids = select_households_by_share(
        household_ids,
        float(active_targets["ev_share_percent"]),
        seed=selection_seed,
        salt="ev",
    )
    heat_pump_ids = select_households_by_share(
        household_ids,
        float(active_targets["heat_pump_share_percent"]),
        seed=selection_seed,
        salt="heat_pump",
    )
    battery_ids = select_households_by_share(
        household_ids,
        float(active_targets["battery_share_percent"]),
        seed=selection_seed,
        salt="battery",
    )
    pv_ids = select_households_by_share(
        household_ids,
        float(active_targets["pv_share_percent"]),
        seed=selection_seed,
        salt="pv",
    )

    return {
        "has_ev": household_id in ev_ids,
        "has_heat_pump": household_id in heat_pump_ids,
        "has_battery": household_id in battery_ids,
        "has_pv": household_id in pv_ids,
    }


def show_household_gridcreator_details(network, household_id: str, original_devices) -> None:
    """Show original GridCreator/grid_model data for the selected household."""
    st.markdown("#### Originaldaten dieses Haushalts aus GridCreator / grid_model")

    load_stats = household_load_profile_stats(network, household_id)
    ev_availability_stats = household_ev_availability_stats(network, household_id)

    rows = [
        {
            "Merkmal": "EV",
            "GridCreator-/grid_model-Wert": yes_no(read_object_value(original_devices, "ev", False)),
            "Detail": optional_power_text(original_devices, "ev_kw", "kW"),
        },
        {
            "Merkmal": "Wärmepumpe",
            "GridCreator-/grid_model-Wert": yes_no(read_object_value(original_devices, "heat_pump", False)),
            "Detail": optional_power_text(original_devices, "heat_pump_kw", "kW"),
        },
        {
            "Merkmal": "Batterie",
            "GridCreator-/grid_model-Wert": yes_no(read_object_value(original_devices, "battery", False)),
            "Detail": optional_power_text(original_devices, "battery_kwh", "kWh"),
        },
        {
            "Merkmal": "PV",
            "GridCreator-/grid_model-Wert": yes_no(read_object_value(original_devices, "pv", False)),
            "Detail": optional_power_text(original_devices, "pv_kwp", "kWp"),
        },
        {
            "Merkmal": "Originales Lastprofil",
            "GridCreator-/grid_model-Wert": "Vorhanden" if load_stats["has_profile"] else "Nicht vorhanden",
            "Detail": load_stats["label"],
        },
        {
            "Merkmal": "EV-Verfügbarkeit",
            "GridCreator-/grid_model-Wert": "Vorhanden" if ev_availability_stats["has_profile"] else "Nicht vorhanden",
            "Detail": ev_availability_stats["label"],
        },
    ]

    st.table(rows)


def grid_model_default_load_scaling_factor(network, household_id: str) -> float:
    """
    Return the default load scaling factor for one household.

    Currently, grid_model provides the real demand as household_load_profile_kw.
    Therefore, the default scaling factor is 1.00, meaning:
    use the generated real load profile unchanged.

    The additional factor is a scenario parameter from the Map UI, not a separate
    original GridCreator/raw grid_model value.
    """
    household_id = str(household_id)

    possible_factor_maps = [
        getattr(network, "household_load_scaling_factor", None),
        getattr(network, "household_load_scaling_factors", None),
        getattr(network, "load_scaling_by_bus", None),
        getattr(network, "default_load_scaling_by_bus", None),
    ]

    for factor_map in possible_factor_maps:
        if isinstance(factor_map, dict) and household_id in factor_map:
            try:
                return float(factor_map[household_id])
            except (TypeError, ValueError):
                pass

    return 1.0


def household_load_profile_stats(network, household_id: str) -> dict[str, Any]:
    """Summarise a household load profile from GridCreator/grid_model."""
    profile = getattr(network, "household_load_profile_kw", {}).get(household_id)

    if not profile:
        return {
            "has_profile": False,
            "label": "Kein Lastprofil vorhanden",
        }

    values = [float(v) for v in profile]
    step_count = len(values)
    dt_hours = 24.0 / step_count if step_count else 0.25
    energy_kwh = sum(values) * dt_hours
    peak_kw = max(values) if values else 0.0
    mean_kw = sum(values) / step_count if step_count else 0.0

    return {
        "has_profile": True,
        "step_count": step_count,
        "energy_kwh": energy_kwh,
        "peak_kw": peak_kw,
        "mean_kw": mean_kw,
        "label": (
            f"{step_count} Zeitschritte, "
            f"Tagesenergie {energy_kwh:.2f} kWh, "
            f"Peak {peak_kw:.2f} kW, "
            f"Durchschnitt {mean_kw:.2f} kW"
        ),
    }


def household_ev_availability_stats(network, household_id: str) -> dict[str, Any]:
    """Summarise EV availability information for one household."""
    availability = getattr(network, "ev_availability", {}).get(household_id)

    if not availability:
        return {
            "has_profile": False,
            "label": "Keine EV-Verfügbarkeit vorhanden",
        }

    connected_steps = sum(1 for value in availability if bool(value))
    total_steps = len(availability)
    share = connected_steps / total_steps * 100.0 if total_steps else 0.0

    return {
        "has_profile": True,
        "connected_steps": connected_steps,
        "total_steps": total_steps,
        "share_percent": share,
        "label": f"{connected_steps}/{total_steps} Zeitschritte verbunden ({share:.1f} %)",
    }


def auto_label(default_value: bool) -> str:
    """Label for automatic values in the individual household configuration."""
    return f"Automatisch (aktuell: {'Ja' if default_value else 'Nein'})"


def auto_help(
    device_label: str,
    uses_gridcreator_defaults: bool,
    global_targets_active: bool,
) -> str:
    """Help text for the automatic individual household option."""
    if uses_gridcreator_defaults and not global_targets_active:
        return f"Automatisch = ursprünglicher GridCreator-Wert für diesen Haushalt ({device_label})."

    if global_targets_active:
        return f"Automatisch = aus den gespeicherten globalen Zielwerten für {device_label} abgeleitet."

    return f"Automatisch = aus der globalen Szenario-Verteilung für {device_label} abgeleitet."


def yes_no(value: Any) -> str:
    """Convert truthy values into a German yes/no label."""
    return "Ja" if bool(value) else "Nein"


def optional_power_text(obj, attr_name: str, unit: str) -> str:
    """Return a formatted optional numeric detail from an object or dictionary."""
    if obj is None:
        return "Nicht vorhanden"

    value = read_object_value(obj, attr_name, None)

    if value is None:
        return "Nicht gespeichert"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    return f"{numeric:.2f} {unit}"


def read_object_value(obj: Any, attr_name: str, default: Any = None) -> Any:
    """Read a value from either a normal object or a dictionary."""
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(attr_name, default)

    return getattr(obj, attr_name, default)