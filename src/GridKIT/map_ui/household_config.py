# ─────────────────────────────────────────────────────────────
# map_ui/household_config.py
#
# Helper functions for configuring existing household/load buses
# after a GridNetwork has been generated.
#
# This module does not change the network topology.
# It only creates a household_configuration.json structure.
#
# Covers all four device types the simulation knows about
# (EV, Batterie, Wärmepumpe, PV) so the configuration can be turned
# straight into a core.models device layout.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import random
from typing import Any

import core.constants as const
from core.models import HouseholdDevices


DEFAULT_EV_SHARE_PERCENT = 30
DEFAULT_HEAT_PUMP_SHARE_PERCENT = 20
DEFAULT_BATTERY_SHARE_PERCENT = 20
DEFAULT_PV_SHARE_PERCENT = 30
DEFAULT_GLOBAL_LOAD_SCALING_FACTOR = 1.0
DEFAULT_SELECTION_SEED = 42

# Device key → (assumption field, override field, resolved field, selection salt).
# Salts keep each device's deterministic draw independent, so raising the EV share
# does not reshuffle which homes got a heat pump.
DEVICE_CONFIG_FIELDS = {
    const.DEVICE_EV: ("ev_share_percent", "has_ev", "ev_bus_ids", "ev"),
    const.DEVICE_BATTERY: ("battery_share_percent", "has_battery", "battery_bus_ids", "battery"),
    const.DEVICE_HEAT_PUMP: ("heat_pump_share_percent", "has_heat_pump", "heat_pump_bus_ids", "heat_pump"),
    const.DEVICE_PV: ("pv_share_percent", "has_pv", "pv_bus_ids", "pv"),
}


def default_scenario_assumptions() -> dict[str, Any]:
    return {
        "ev_share_percent": DEFAULT_EV_SHARE_PERCENT,
        "battery_share_percent": DEFAULT_BATTERY_SHARE_PERCENT,
        "heat_pump_share_percent": DEFAULT_HEAT_PUMP_SHARE_PERCENT,
        "pv_share_percent": DEFAULT_PV_SHARE_PERCENT,
        "global_load_scaling_factor": DEFAULT_GLOBAL_LOAD_SCALING_FACTOR,
        "selection_seed": DEFAULT_SELECTION_SEED,
    }


def choice_index_from_bool(value: bool | None) -> int:
    if value is True:
        return 1
    if value is False:
        return 2
    return 0


def bool_from_choice(choice: str) -> bool | None:
    if choice == "Ja":
        return True
    if choice == "Nein":
        return False
    return None


def select_households_by_share(
    household_ids: list[str],
    share_percent: int,
    seed: int,
    salt: str,
) -> list[str]:
    if not household_ids or share_percent <= 0:
        return []

    if share_percent >= 100:
        return sorted(household_ids)

    count = int(len(household_ids) * (share_percent / 100.0) + 0.5)
    count = max(0, min(count, len(household_ids)))

    shuffled_ids = list(household_ids)
    rng = random.Random(f"{seed}-{salt}")
    rng.shuffle(shuffled_ids)

    return sorted(shuffled_ids[:count])


def build_household_configuration(
    network: Any,
    selected_bounds: Any,
    household_ids: list[str],
    ev_share_percent: int,
    heat_pump_share_percent: int,
    global_load_scaling_factor: float,
    selection_seed: int,
    household_overrides: dict[str, dict[str, Any]],
    battery_share_percent: int = DEFAULT_BATTERY_SHARE_PERCENT,
    pv_share_percent: int = DEFAULT_PV_SHARE_PERCENT,
) -> dict[str, Any]:
    share_by_device = {
        const.DEVICE_EV: int(ev_share_percent),
        const.DEVICE_BATTERY: int(battery_share_percent),
        const.DEVICE_HEAT_PUMP: int(heat_pump_share_percent),
        const.DEVICE_PV: int(pv_share_percent),
    }

    resolved_by_device: dict[str, set[str]] = {}
    for device, (_, _, _, salt) in DEVICE_CONFIG_FIELDS.items():
        resolved_by_device[device] = set(
            select_households_by_share(
                household_ids=household_ids,
                share_percent=share_by_device[device],
                seed=selection_seed,
                salt=salt,
            )
        )

    resolved_load_scaling_by_bus = {
        bus_id: float(global_load_scaling_factor)
        for bus_id in household_ids
    }

    cleaned_overrides: dict[str, dict[str, Any]] = {}

    for bus_id, override in household_overrides.items():
        if bus_id not in household_ids:
            continue

        cleaned_override: dict[str, Any] = {}

        for device, (_, override_field, _, _) in DEVICE_CONFIG_FIELDS.items():
            if override_field in override:
                has_device = bool(override[override_field])
                cleaned_override[override_field] = has_device

                if has_device:
                    resolved_by_device[device].add(bus_id)
                else:
                    resolved_by_device[device].discard(bus_id)

        if "load_scaling_factor" in override:
            load_scaling_factor = float(override["load_scaling_factor"])
            cleaned_override["load_scaling_factor"] = load_scaling_factor
            resolved_load_scaling_by_bus[bus_id] = load_scaling_factor

        if cleaned_override:
            cleaned_overrides[bus_id] = cleaned_override

    resolved: dict[str, Any] = {
        resolved_field: sorted(resolved_by_device[device])
        for device, (_, _, resolved_field, _) in DEVICE_CONFIG_FIELDS.items()
    }
    resolved["load_scaling_by_bus"] = resolved_load_scaling_by_bus

    return {
        "configuration_type": "household_scenario_configuration",
        "version": 2,
        "topology_changed": False,
        "description": (
            "Konfiguration von Szenario-Annahmen und individuellen Anpassungen "
            "für vorhandene Haushalts-/Last-Busse."
        ),
        "network": {
            "network_id": network.network_id,
            "area_name": getattr(network, "area_name", None),
            "bbox": selected_bounds.model_dump(),
            "area_km2": selected_bounds.approx_area_km2(),
            "household_count": len(household_ids),
        },
        "scenario_assumptions": {
            "ev_share_percent": int(ev_share_percent),
            "battery_share_percent": int(battery_share_percent),
            "heat_pump_share_percent": int(heat_pump_share_percent),
            "pv_share_percent": int(pv_share_percent),
            "global_load_scaling_factor": float(global_load_scaling_factor),
            "selection_seed": int(selection_seed),
            "selection_method": (
                "Deterministische zufällige Auswahl aus vorhandenen household_bus_ids. "
                "Individuelle Haushalt-Anpassungen überschreiben diese Szenario-Annahmen."
            ),
            "note": (
                "Diese Werte werden nicht automatisch aus OSM erkannt, "
                "sondern als Annahmen für das Szenario verwendet."
            ),
        },
        "individual_household_adjustments": cleaned_overrides,
        "resolved": resolved,
        "assumptions": {
            "ev_means": "Electric Vehicle / Elektrofahrzeug am Haushalt",
            "battery_means": "Heim-Batteriespeicher am Haushalt",
            "heat_pump_means": "Wärmepumpe am Haushalt",
            "pv_means": "PV-Anlage auf dem Haushalt",
            "load_scaling_factor_means": (
                "1.0 = unverändertes Lastprofil, 1.2 = 20 Prozent höherer Verbrauch"
            ),
            "topology_note": "Die Netztopologie des erzeugten GridNetwork bleibt unverändert.",
        },
    }


def device_layout_from_configuration(
    household_ids: list[str],
    configuration: dict[str, Any],
) -> dict[str, HouseholdDevices]:
    """Turn a household_configuration into the per-home layout GridEnv consumes.

    This is the bridge between the map UI's configuration format and
    core.models — the RL environment and the dashboard both work off
    HouseholdDevices, never off the JSON structure.
    """
    resolved = configuration["resolved"]
    equipped = {
        device: set(resolved.get(resolved_field, []))
        for device, (_, _, resolved_field, _) in DEVICE_CONFIG_FIELDS.items()
    }

    return {
        bus_id: HouseholdDevices(
            bus_id=bus_id,
            ev=bus_id in equipped[const.DEVICE_EV],
            battery=bus_id in equipped[const.DEVICE_BATTERY],
            heat_pump=bus_id in equipped[const.DEVICE_HEAT_PUMP],
            pv=bus_id in equipped[const.DEVICE_PV],
        )
        for bus_id in household_ids
    }


def json_dumps_pretty(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
