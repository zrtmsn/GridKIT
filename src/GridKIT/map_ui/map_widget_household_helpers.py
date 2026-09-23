# ─────────────────────────────────────────────────────────────
# map_ui/map_widget_household_helpers.py
#
# Helper functions for map_widget_households.py.
#
# This file contains:
# - household labels and sorting
# - GridCreator/device summaries
# - automatic/global household override logic
# - household load profile and EV availability summaries
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from typing import Any

import streamlit as st

from map_ui.household_config import select_households_by_share
from map_ui.transformer_network_filter import (
    assigned_household_ids_for_transformer,
    device_summary,
    selectable_transformer_ids,
)


LOAD_PROFILE_MODE_ORIGINAL = "Originales Lastprofil unverändert verwenden"
LOAD_PROFILE_MODE_GLOBAL = "Globale Szenario-Skalierung anwenden"
LOAD_PROFILE_MODE_CUSTOM = "Eigene Skalierung für diesen Haushalt festlegen"

MIN_PERCENT = 0.0
MAX_PERCENT = 100.0
PERCENT_SLIDER_DECIMALS = 1

DEVICE_METRIC_COLUMN_COUNT = 5

DEFAULT_LOAD_SCALING_FACTOR = 1.0
HOURS_PER_DAY = 24.0
DEFAULT_TIMESTEP_HOURS = 0.25

DISPLAY_DECIMALS = 2
EV_AVAILABILITY_DECIMALS = 1

UNKNOWN_REFERENCE_POSITION = 10**12
UNKNOWN_BUILDING_SORT_NUMBER = 10**18

SORT_GROUP_WITH_BUILDING_ID = 0
SORT_GROUP_WITHOUT_BUILDING_ID = 1

HOUSEHOLD_NUMBER_OFFSET = 1
BUILDING_ID_MARKER = "_building_"


def baseline_device_targets(
    network,
    saved_assumptions: dict[str, Any],
    uses_gridcreator_defaults: bool,
) -> dict[str, float]:
    """Return the default global target shares for the displayed network."""
    if uses_gridcreator_defaults:
        summary = device_summary(network)

        return {
            "ev_share_percent": float(summary["ev_share_percent"]),
            "heat_pump_share_percent": float(summary["heat_pump_share_percent"]),
            "battery_share_percent": float(summary["battery_share_percent"]),
            "pv_share_percent": float(summary["pv_share_percent"]),
        }

    return {
        "ev_share_percent": float(saved_assumptions.get("ev_share_percent", MIN_PERCENT)),
        "heat_pump_share_percent": float(
            saved_assumptions.get("heat_pump_share_percent", MIN_PERCENT)
        ),
        "battery_share_percent": float(saved_assumptions.get("battery_share_percent", MIN_PERCENT)),
        "pv_share_percent": float(saved_assumptions.get("pv_share_percent", MIN_PERCENT)),
    }


def percent_slider_value(value: Any) -> float:
    """Return a bounded percentage value for Streamlit sliders."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = MIN_PERCENT

    return max(MIN_PERCENT, min(MAX_PERCENT, round(numeric, PERCENT_SLIDER_DECIMALS)))


def show_gridcreator_device_metrics(summary: dict[str, Any]) -> None:
    """Show the device summary for the currently displayed network."""
    st.markdown("#### Erkannte Ausstattung der Haushalte")

    c1, c2, c3, c4, c5 = st.columns(DEVICE_METRIC_COLUMN_COUNT)
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
    """
    Combine global target shares with individual household overrides.

    Individual overrides have the highest priority. Global overrides are
    generated only when saved global targets exist or no real GridCreator
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
    """
    Return the values used when an individual setting stays automatic.
    """
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


def show_household_gridcreator_details(
    network,
    household_id: str,
    household_label: str,
    original_devices,
) -> None:
    """Show original network/model data for the selected household."""
    st.markdown(f"#### Ausgangsdaten: {household_label}")

    load_stats = household_load_profile_stats(network, household_id)
    ev_availability_stats = household_ev_availability_stats(network, household_id)

    rows = [
        {
            "Merkmal": "EV",
            "Originalwert": yes_no(read_object_value(original_devices, "ev", False)),
            "Detail": optional_power_text(original_devices, "ev_kw", "kW"),
        },
        {
            "Merkmal": "Wärmepumpe",
            "Originalwert": yes_no(read_object_value(original_devices, "heat_pump", False)),
            "Detail": optional_power_text(original_devices, "heat_pump_kw", "kW"),
        },
        {
            "Merkmal": "Batterie",
            "Originalwert": yes_no(read_object_value(original_devices, "battery", False)),
            "Detail": optional_power_text(original_devices, "battery_kwh", "kWh"),
        },
        {
            "Merkmal": "PV",
            "Originalwert": yes_no(read_object_value(original_devices, "pv", False)),
            "Detail": optional_power_text(original_devices, "pv_kwp", "kWp"),
        },
        {
            "Merkmal": "Originales Lastprofil",
            "Originalwert": "Vorhanden" if load_stats["has_profile"] else "Nicht vorhanden",
            "Detail": load_stats["label"],
        },
        {
            "Merkmal": "EV-Verfügbarkeit",
            "Originalwert": "Vorhanden" if ev_availability_stats["has_profile"] else "Nicht vorhanden",
            "Detail": ev_availability_stats["label"],
        },
    ]

    st.table(rows)


def grid_model_default_load_scaling_factor(network, household_id: str) -> float:
    """
    Return the default load scaling factor for one household.

    grid_model provides the real demand as household_load_profile_kw. Therefore,
    the default scaling factor is 1.0, which keeps the generated profile unchanged.
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

    return DEFAULT_LOAD_SCALING_FACTOR


def household_load_profile_stats(network, household_id: str) -> dict[str, Any]:
    """Summarize a household load profile from GridCreator/grid_model."""
    profile = getattr(network, "household_load_profile_kw", {}).get(household_id)

    if not profile:
        return {
            "has_profile": False,
            "label": "Kein Lastprofil vorhanden",
        }

    values = [float(v) for v in profile]
    step_count = len(values)
    dt_hours = HOURS_PER_DAY / step_count if step_count else DEFAULT_TIMESTEP_HOURS
    energy_kwh = sum(values) * dt_hours
    peak_kw = max(values) if values else MIN_PERCENT
    mean_kw = sum(values) / step_count if step_count else MIN_PERCENT

    return {
        "has_profile": True,
        "step_count": step_count,
        "energy_kwh": energy_kwh,
        "peak_kw": peak_kw,
        "mean_kw": mean_kw,
        "label": (
            f"{step_count} Zeitschritte, "
            f"Tagesenergie {energy_kwh:.{DISPLAY_DECIMALS}f} kWh, "
            f"Peak {peak_kw:.{DISPLAY_DECIMALS}f} kW, "
            f"Durchschnitt {mean_kw:.{DISPLAY_DECIMALS}f} kW"
        ),
    }


def household_ev_availability_stats(network, household_id: str) -> dict[str, Any]:
    """Summarize EV availability information for one household."""
    availability = getattr(network, "ev_availability", {}).get(household_id)

    if not availability:
        return {
            "has_profile": False,
            "label": "Keine EV-Verfügbarkeit vorhanden",
        }

    connected_steps = sum(1 for value in availability if bool(value))
    total_steps = len(availability)
    share = connected_steps / total_steps * MAX_PERCENT if total_steps else MIN_PERCENT

    return {
        "has_profile": True,
        "connected_steps": connected_steps,
        "total_steps": total_steps,
        "share_percent": share,
        "label": (
            f"{connected_steps}/{total_steps} Zeitschritte verbunden "
            f"({share:.{EV_AVAILABILITY_DECIMALS}f} %)"
        ),
    }


def auto_label(default_value: bool) -> str:
    """Return the label for automatic individual household values."""
    return f"Automatisch (aktuell: {'Ja' if default_value else 'Nein'})"


def auto_help(
    device_label: str,
    uses_gridcreator_defaults: bool,
    global_targets_active: bool,
) -> str:
    """Return help text for automatic individual household values."""
    if uses_gridcreator_defaults and not global_targets_active:
        return f"Automatisch = ursprünglicher Wert für diesen Haushalt ({device_label})."

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

    return f"{numeric:.{DISPLAY_DECIMALS}f} {unit}"


def household_label_reference_order(network) -> list[str]:
    """
    Return household IDs in a stable, user-friendly order.

    The numbering follows the order of the user-facing transformer areas. The
    full network is used when available, so numbering stays stable under filters.
    """
    full_network = st.session_state.get("full_network") or network

    all_household_ids = {
        str(bus_id)
        for bus_id in getattr(full_network, "household_bus_ids", [])
    }

    if not all_household_ids:
        return sorted(
            str(bus_id)
            for bus_id in getattr(network, "household_bus_ids", [])
        )

    ordered_household_ids: list[str] = []
    already_added: set[str] = set()

    for trafo_id in selectable_transformer_ids(full_network):
        try:
            households_in_area = assigned_household_ids_for_transformer(
                full_network,
                trafo_id,
            )
        except Exception:
            continue

        households_in_area_ordered = [
            household_id
            for household_id in all_household_ids
            if household_id in households_in_area
            and household_id not in already_added
        ]

        households_in_area_ordered.sort(key=household_sort_key)

        for household_id in households_in_area_ordered:
            ordered_household_ids.append(household_id)
            already_added.add(household_id)

    remaining_households = [
        household_id
        for household_id in all_household_ids
        if household_id not in already_added
    ]

    remaining_households.sort(key=household_sort_key)
    ordered_household_ids.extend(remaining_households)

    return ordered_household_ids


def sort_household_ids_for_display(
    household_ids: list[str],
    reference_household_ids: list[str],
) -> list[str]:
    """Sort visible household IDs according to the global reference order."""
    reference_position = {
        str(household_id): index
        for index, household_id in enumerate(reference_household_ids)
    }

    return sorted(
        (str(household_id) for household_id in household_ids),
        key=lambda household_id: (
            reference_position.get(household_id, UNKNOWN_REFERENCE_POSITION),
            household_sort_key(household_id),
        ),
    )


def household_display_label(household_ids: list[str], household_id: str) -> str:
    """Return a user-friendly household label while keeping the internal ID."""
    household_id = str(household_id)
    household_ids_in_order = [str(item) for item in household_ids]

    try:
        household_number = (
            household_ids_in_order.index(household_id)
            + HOUSEHOLD_NUMBER_OFFSET
        )
    except ValueError:
        household_number = None

    building_id = extract_building_id(household_id)

    if household_number is None:
        if building_id:
            return f"Haushalt – Gebäude {building_id}"

        return "Haushalt"

    if building_id:
        return f"Haushalt {household_number} – Gebäude {building_id}"

    return f"Haushalt {household_number}"


def household_sort_key(household_id: str) -> tuple[int, int, str]:
    """Sort households by building number if available."""
    building_id = extract_building_id(household_id)

    if building_id is not None:
        try:
            return SORT_GROUP_WITH_BUILDING_ID, int(building_id), str(household_id)
        except ValueError:
            pass

    return SORT_GROUP_WITHOUT_BUILDING_ID, UNKNOWN_BUILDING_SORT_NUMBER, str(household_id)


def extract_building_id(bus_id: str) -> str | None:
    """Extract a building identifier from known household bus IDs."""
    bus_id = str(bus_id)

    if BUILDING_ID_MARKER in bus_id:
        building_id = bus_id.rsplit(BUILDING_ID_MARKER, 1)[-1].strip()
        return building_id or None

    numeric_parts = re.findall(r"\d+", bus_id)

    if numeric_parts:
        return numeric_parts[-1]

    return None


def read_object_value(obj: Any, attr_name: str, default: Any = None) -> Any:
    """Read a value from either a normal object or a dictionary."""
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(attr_name, default)

    return getattr(obj, attr_name, default)