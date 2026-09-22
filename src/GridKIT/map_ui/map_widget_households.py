# ─────────────────────────────────────────────────────────────
# map_ui/map_widget_households.py
#
# Household configuration UI for map_widget.py.
#
# This file contains the main Streamlit UI flow:
# - global target sliders
# - individual household overrides
# - household_configuration.json creation
#
# Helper functions for labels, sorting, summaries and model details live in
# map_ui/map_widget_household_helpers.py.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

import streamlit as st

from map_ui.area_bounds import AreaBounds
from map_ui.household_config import (
    DEFAULT_GLOBAL_LOAD_SCALING_FACTOR,
    DEFAULT_SELECTION_SEED,
    MAX_SHARE_PERCENT,
    MIN_SHARE_PERCENT,
    bool_from_choice,
    build_household_configuration,
    choice_index_from_bool,
    default_scenario_assumptions,
    gridcreator_defaults,
    json_dumps_pretty,
)
from map_ui.map_widget_household_helpers import (
    LOAD_PROFILE_MODE_CUSTOM,
    LOAD_PROFILE_MODE_GLOBAL,
    LOAD_PROFILE_MODE_ORIGINAL,
    automatic_device_values_for_household,
    auto_help,
    auto_label,
    baseline_device_targets,
    build_effective_household_overrides,
    grid_model_default_load_scaling_factor,
    household_display_label,
    household_label_reference_order,
    percent_slider_value,
    show_gridcreator_device_metrics,
    show_household_gridcreator_details,
    sort_household_ids_for_display,
)
from map_ui.transformer_network_filter import device_summary


TARGET_SLIDER_STEP = 0.1
TARGET_SLIDER_FORMAT = "%.1f"

LOAD_SCALING_MIN = 0.1
LOAD_SCALING_MAX = 5.0
LOAD_SCALING_STEP = 0.1
LOAD_SCALING_FORMAT = "%.2f"

SELECTION_SEED_MIN = 0
SELECTION_SEED_MAX = 99_999
SELECTION_SEED_STEP = 1

GLOBAL_TARGET_COLUMN_COUNT = 4
ACTION_COLUMN_COUNT = 2
HOUSEHOLD_SETTING_COLUMN_COUNT = 5
SUMMARY_COLUMN_COUNT = 6

FLOAT_COMPARISON_TOLERANCE = 1e-9


def show_household_configuration(network, selected_bounds: AreaBounds) -> dict[str, Any] | None:
    """
    Render the household configuration UI for the currently displayed network.

    The UI can operate on the complete generated network or on a selected
    transformer area, depending on the active transformer filter.
    """
    st.subheader("Haushaltskonfiguration")

    selected_trafo_id = st.session_state.get("selected_trafo_id")
    raw_household_ids = sorted(str(bus_id) for bus_id in getattr(network, "household_bus_ids", []))
    reference_household_ids = household_label_reference_order(network)
    household_ids = sort_household_ids_for_display(raw_household_ids, reference_household_ids)

    if selected_trafo_id is None:
        st.info(
            "Es ist aktuell kein Trafo-Filter aktiv. "
            "Die Haushaltskonfiguration bezieht sich auf alle Haushalte "
            "im vollständig erzeugten Netz des ausgewählten Kartenbereichs."
        )
    else:
        st.info(
            "Die Haushaltskonfiguration bezieht sich auf den aktuell ausgewählten "
            "Transformatorbereich. Es werden nur die Haushalte angezeigt und "
            "konfiguriert, die zu diesem Bereich gehören."
        )

    if not household_ids:
        if selected_trafo_id is None:
            st.warning(
                "Im erzeugten Netz wurden keine Haushalte gefunden. "
                "Die Haushaltskonfiguration kann deshalb nicht angewendet werden."
            )
        else:
            st.warning(
                "Im ausgewählten Transformatorbereich wurden keine Haushalte gefunden. "
                "Die Haushaltskonfiguration kann deshalb nicht angewendet werden."
            )
        return None

    if st.session_state.pop("scenario_assumptions_saved_message", False):
        st.success("Konfiguration wurde gespeichert.")

    if st.session_state.pop("scenario_assumptions_reset_message", False):
        st.success("Konfiguration wurde auf die Ausgangsdaten zurückgesetzt.")

    saved_household = st.session_state.pop("household_saved_message", None)
    if saved_household:
        st.success(f"Individuelle Anpassung für {saved_household} gespeichert.")

    reset_household = st.session_state.pop("household_reset_message", None)
    if reset_household:
        st.info(f"Individuelle Anpassung für {reset_household} wurde gelöscht.")

    # Session state stores scenario settings across Streamlit reruns.
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

    # A new scope invalidates previously saved global target values.
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

    st.markdown("### Ausgangsdaten des Netzmodells")

    if not uses_gridcreator_defaults:
        st.info(
            "Für dieses Netz liegen keine vollständigen Gerätedaten pro Haushalt vor. "
            "Die Ausstattung wird deshalb über globale Zielwerte verteilt. "
            "Sobald vollständige Gerätedaten vorhanden sind, werden diese automatisch "
            "als Ausgangszustand verwendet."
        )

    show_gridcreator_device_metrics(gridcreator_summary)

    st.caption(
        "Diese Werte stammen aus dem erzeugten Netzmodell und dienen als Grundlage "
        "für die folgenden Szenario-Anpassungen."
    )

    st.divider()

    st.markdown("### Szenario-Anpassungen")

    st.info(
        "Die folgenden Zielwerte sind Szenario-Annahmen. "
        "Nach dem Speichern werden sie verwendet, um die Haushaltsausstattung "
        "und die Lastprofile für das aktuelle Szenario anzupassen."
    )

    st.caption(
        "Für jeden Haushalt liegt ein ursprüngliches Lastprofil vor. "
        "Eine Skalierung von 1.00 bedeutet, dass dieses Profil unverändert verwendet wird. "
        "Höhere oder niedrigere Werte dienen als Szenario-Annahme, um den Verbrauch zu verändern."
    )

    widget_suffix = st.session_state["scenario_config_version"]

    high_col1, high_col2, high_col3, high_col4 = st.columns(GLOBAL_TARGET_COLUMN_COUNT)

    draft_ev_share_percent = high_col1.slider(
        "Ziel-Anteil Haushalte mit EV (%)",
        min_value=MIN_SHARE_PERCENT,
        max_value=MAX_SHARE_PERCENT,
        value=percent_slider_value(active_targets["ev_share_percent"]),
        step=TARGET_SLIDER_STEP,
        format=TARGET_SLIDER_FORMAT,
        help=(
            "Ausgangswert ist der erkannte EV-Anteil im ausgewählten Netz. "
            "Nach dem Speichern kann dieser Zielwert für das Szenario geändert werden."
        ),
        key=f"draft_ev_share_percent_{widget_suffix}",
    )

    draft_heat_pump_share_percent = high_col2.slider(
        "Ziel-Anteil Haushalte mit WP (%)",
        min_value=MIN_SHARE_PERCENT,
        max_value=MAX_SHARE_PERCENT,
        value=percent_slider_value(active_targets["heat_pump_share_percent"]),
        step=TARGET_SLIDER_STEP,
        format=TARGET_SLIDER_FORMAT,
        help=(
            "Ausgangswert ist der erkannte Wärmepumpen-Anteil im ausgewählten Netz. "
            "Nach dem Speichern kann dieser Zielwert für das Szenario geändert werden."
        ),
        key=f"draft_heat_pump_share_percent_{widget_suffix}",
    )

    draft_battery_share_percent = high_col3.slider(
        "Ziel-Anteil Haushalte mit Batterie (%)",
        min_value=MIN_SHARE_PERCENT,
        max_value=MAX_SHARE_PERCENT,
        value=percent_slider_value(active_targets["battery_share_percent"]),
        step=TARGET_SLIDER_STEP,
        format=TARGET_SLIDER_FORMAT,
        help=(
            "Ausgangswert ist der erkannte Batterie-Anteil im ausgewählten Netz. "
            "Nach dem Speichern kann dieser Zielwert für das Szenario geändert werden."
        ),
        key=f"draft_battery_share_percent_{widget_suffix}",
    )

    draft_pv_share_percent = high_col4.slider(
        "Ziel-Anteil Haushalte mit PV (%)",
        min_value=MIN_SHARE_PERCENT,
        max_value=MAX_SHARE_PERCENT,
        value=percent_slider_value(active_targets["pv_share_percent"]),
        step=TARGET_SLIDER_STEP,
        format=TARGET_SLIDER_FORMAT,
        help=(
            "Ausgangswert ist der erkannte PV-Anteil im ausgewählten Netz. "
            "Nach dem Speichern kann dieser Zielwert für das Szenario geändert werden."
        ),
        key=f"draft_pv_share_percent_{widget_suffix}",
    )

    draft_global_load_scaling_factor = st.number_input(
        "Globale Lastprofil-Skalierung für Szenarien",
        min_value=LOAD_SCALING_MIN,
        max_value=LOAD_SCALING_MAX,
        value=float(
            saved_assumptions.get(
                "global_load_scaling_factor",
                DEFAULT_GLOBAL_LOAD_SCALING_FACTOR,
            )
        ),
        step=LOAD_SCALING_STEP,
        format=LOAD_SCALING_FORMAT,
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
            min_value=SELECTION_SEED_MIN,
            max_value=SELECTION_SEED_MAX,
            value=int(saved_assumptions.get("selection_seed", DEFAULT_SELECTION_SEED)),
            step=SELECTION_SEED_STEP,
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

    action_col1, action_col2 = st.columns(ACTION_COLUMN_COUNT)

    if action_col1.button("Szenario-Anpassungen speichern"):
        st.session_state["global_device_targets"] = draft_device_targets
        st.session_state["scenario_assumptions"] = draft_assumptions
        st.session_state["scenario_config_version"] += 1
        st.session_state["scenario_assumptions_saved_message"] = True
        st.rerun()

    if action_col2.button("Auf Ausgangsdaten zurücksetzen"):
        reset_assumptions = default_scenario_assumptions()
        reset_assumptions["global_load_scaling_factor"] = DEFAULT_GLOBAL_LOAD_SCALING_FACTOR
        reset_assumptions["selection_seed"] = int(
            saved_assumptions.get("selection_seed", DEFAULT_SELECTION_SEED)
        )

        st.session_state["global_device_targets"] = None
        st.session_state["scenario_assumptions"] = reset_assumptions
        st.session_state["scenario_config_version"] += 1
        st.session_state["scenario_assumptions_reset_message"] = True
        st.rerun()

    if global_targets_active:
        st.warning(
            "Es sind gespeicherte globale Zielwerte aktiv. "
            "Die aktuelle Haushaltskonfiguration basiert deshalb nicht mehr ausschließlich "
            "auf den ursprünglichen Gerätedaten."
        )

    if (
        draft_device_targets != active_targets
        or float(draft_global_load_scaling_factor)
        != float(
            saved_assumptions.get(
                "global_load_scaling_factor",
                DEFAULT_GLOBAL_LOAD_SCALING_FACTOR,
            )
        )
        or int(draft_selection_seed)
        != int(saved_assumptions.get("selection_seed", DEFAULT_SELECTION_SEED))
    ):
        st.warning(
            "Es gibt nicht gespeicherte Änderungen in der globalen Konfiguration. "
            "Klicke auf „Szenario-Anpassungen speichern“, damit diese Werte übernommen werden."
        )

    saved_global_load_scaling_factor = float(
        saved_assumptions.get(
            "global_load_scaling_factor",
            DEFAULT_GLOBAL_LOAD_SCALING_FACTOR,
        )
    )
    saved_selection_seed = int(saved_assumptions.get("selection_seed", DEFAULT_SELECTION_SEED))

    effective_overrides = build_effective_household_overrides(
        household_ids=household_ids,
        device_targets=active_targets,
        selection_seed=saved_selection_seed,
        individual_overrides=individual_overrides,
        generate_global_overrides=global_targets_active or not uses_gridcreator_defaults,
    )

    st.markdown("### Individuelle Anpassung einzelner Haushalte")

    selected_household = st.selectbox(
        "Haushalt auswählen",
        household_ids,
        format_func=lambda household_id: household_display_label(
            reference_household_ids,
            household_id,
        ),
        help=(
            "Hier kann ein konkreter Haushalt individuell angepasst werden. "
            "Die technische interne ID bleibt unverändert, wird aber in der UI vereinfacht dargestellt."
        ),
    )

    selected_household_label = household_display_label(reference_household_ids, selected_household)

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
        household_label=selected_household_label,
        original_devices=original_devices,
    )

    original_load_scaling_factor = grid_model_default_load_scaling_factor(
        network=network,
        household_id=selected_household,
    )

    low_col1, low_col2, low_col3, low_col4, low_col5 = st.columns(
        HOUSEHOLD_SETTING_COLUMN_COUNT
    )

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

        if (
            abs(current_load_scaling_override - original_load_scaling_factor)
            < FLOAT_COMPARISON_TOLERANCE
        ):
            load_profile_mode_index = 0
        elif (
            abs(current_load_scaling_override - saved_global_load_scaling_factor)
            < FLOAT_COMPARISON_TOLERANCE
        ):
            load_profile_mode_index = 1
        else:
            load_profile_mode_index = 2

    load_profile_mode = low_col5.selectbox(
        "Lastprofil für diesen Haushalt",
        options=load_profile_mode_options,
        index=load_profile_mode_index,
        key=f"load_profile_mode_{selected_household}_{household_widget_suffix}",
        help=(
            "Originales Lastprofil bedeutet: Die Zeitreihe aus dem Netzmodell "
            "wird unverändert verwendet. Die globale Szenario-Skalierung wendet den "
            f"aktuellen globalen Faktor {saved_global_load_scaling_factor:.2f} an. "
            "Eine eigene Skalierung speichert einen individuellen Faktor nur für diesen Haushalt."
        ),
    )

    configured_load_scaling_factor = low_col5.number_input(
        "Eigene Skalierung",
        min_value=LOAD_SCALING_MIN,
        max_value=LOAD_SCALING_MAX,
        value=float(
            current_override.get(
                "load_scaling_factor",
                original_load_scaling_factor,
            )
        ),
        step=LOAD_SCALING_STEP,
        format=LOAD_SCALING_FORMAT,
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

    individual_action_col1, individual_action_col2 = st.columns(ACTION_COLUMN_COUNT)

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
            if (
                abs(original_load_scaling_factor - saved_global_load_scaling_factor)
                > FLOAT_COMPARISON_TOLERANCE
            ):
                new_override["load_scaling_factor"] = float(original_load_scaling_factor)

        elif load_profile_mode == LOAD_PROFILE_MODE_CUSTOM:
            new_override["load_scaling_factor"] = float(configured_load_scaling_factor)

        if new_override:
            individual_overrides[selected_household] = new_override
            st.session_state["household_config_version"] += 1
            st.session_state["household_saved_message"] = selected_household_label
            st.rerun()
        else:
            individual_overrides.pop(selected_household, None)
            st.session_state["household_config_version"] += 1
            st.session_state["household_reset_message"] = selected_household_label
            st.rerun()

    if individual_action_col2.button(
        "Individuelle Anpassung löschen",
        key=f"delete_override_{selected_household}",
    ):
        individual_overrides.pop(selected_household, None)
        st.session_state["household_config_version"] += 1
        st.session_state["household_reset_message"] = selected_household_label
        st.rerun()

    household_configuration = build_household_configuration(
        network=network,
        selected_bounds=selected_bounds,
        household_ids=household_ids,
        ev_share_percent=float(active_targets["ev_share_percent"]),
        heat_pump_share_percent=float(active_targets["heat_pump_share_percent"]),
        battery_share_percent=float(active_targets["battery_share_percent"]),
        pv_share_percent=float(active_targets["pv_share_percent"]),
        global_load_scaling_factor=saved_global_load_scaling_factor,
        selection_seed=saved_selection_seed,
        household_overrides=effective_overrides,
    )

    st.markdown("### Zusammenfassung der aktuellen Haushaltskonfiguration")

    resolved = household_configuration["resolved"]

    (
        summary_col1,
        summary_col2,
        summary_col3,
        summary_col4,
        summary_col5,
        summary_col6,
    ) = st.columns(SUMMARY_COLUMN_COUNT)

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