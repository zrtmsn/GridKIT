# ─────────────────────────────────────────────────────────────
# map_ui/household_config.py
#
# Helper functions for configuring existing household/load buses
# after a GridNetwork has been generated.
#
# This module does not change the network topology.
# It only creates a household_configuration.json structure.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import random
from typing import Any


DEFAULT_EV_SHARE_PERCENT = 30.0
DEFAULT_HEAT_PUMP_SHARE_PERCENT = 20.0
DEFAULT_BATTERY_SHARE_PERCENT = 0.0
DEFAULT_PV_SHARE_PERCENT = 0.0
DEFAULT_GLOBAL_LOAD_SCALING_FACTOR = 1.0
DEFAULT_SELECTION_SEED = 42


def default_scenario_assumptions() -> dict[str, Any]:
    return {
        "ev_share_percent": DEFAULT_EV_SHARE_PERCENT,
        "heat_pump_share_percent": DEFAULT_HEAT_PUMP_SHARE_PERCENT,
        "battery_share_percent": DEFAULT_BATTERY_SHARE_PERCENT,
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
    share_percent: float,
    seed: int,
    salt: str,
) -> list[str]:
    """
    Select a deterministic subset of households based on a percentage.

    share_percent can be a decimal value, for example:
    - 16.7
    - 33.3
    - 66.7

    The selected count is rounded to the nearest household count.
    """
    normalized_share = normalize_share_percent(share_percent)

    if not household_ids or normalized_share <= 0.0:
        return []

    if normalized_share >= 100.0:
        return sorted(household_ids)

    count = int(len(household_ids) * (normalized_share / 100.0) + 0.5)
    count = max(0, min(count, len(household_ids)))

    shuffled_ids = list(household_ids)
    rng = random.Random(f"{seed}-{salt}")
    rng.shuffle(shuffled_ids)

    return sorted(shuffled_ids[:count])


def gridcreator_defaults(network: Any) -> dict[str, Any]:
    """
    Return the per-household device assignment from GridCreator.

    OSMNetworkBuilder writes real GridCreator device data into
    network.household_devices. If this dictionary is empty, the network does not
    carry real per-household device assignments and the map UI falls back to
    scenario-based assumptions.
    """
    return getattr(network, "household_devices", {}) or {}


def build_household_configuration(
    network: Any,
    selected_bounds: Any,
    household_ids: list[str],
    ev_share_percent: float,
    heat_pump_share_percent: float,
    global_load_scaling_factor: float,
    selection_seed: int,
    household_overrides: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the household configuration JSON for the currently displayed network.

    Resolution order:
    1. real GridCreator device data from network.household_devices
    2. deterministic scenario shares if no GridCreator device data exists
    3. explicit overrides from the map UI

    In the current map UI, household_overrides can contain two kinds of changes:
    - generated global target overrides, created from the global target sliders
    - individual household overrides, created by the per-household UI

    This function treats both as explicit overrides and applies them after the
    GridCreator baseline.
    """
    household_ids = sorted(str(bus_id) for bus_id in household_ids)
    household_id_set = set(household_ids)

    ev_share_percent = normalize_share_percent(ev_share_percent)
    heat_pump_share_percent = normalize_share_percent(heat_pump_share_percent)

    household_devices = {
        str(bus_id): devices
        for bus_id, devices in gridcreator_defaults(network).items()
        if str(bus_id) in household_id_set
    }
    uses_gridcreator_defaults = bool(household_devices)

    if uses_gridcreator_defaults:
        resolved_ev_ids = {
            bus_id
            for bus_id in household_ids
            if bool(read_object_value(household_devices.get(bus_id), "ev", False))
        }
        resolved_heat_pump_ids = {
            bus_id
            for bus_id in household_ids
            if bool(read_object_value(household_devices.get(bus_id), "heat_pump", False))
        }
        resolved_battery_ids = {
            bus_id
            for bus_id in household_ids
            if bool(read_object_value(household_devices.get(bus_id), "battery", False))
        }
        resolved_pv_ids = {
            bus_id
            for bus_id in household_ids
            if bool(read_object_value(household_devices.get(bus_id), "pv", False))
        }
    else:
        resolved_ev_ids = set(
            select_households_by_share(
                household_ids=household_ids,
                share_percent=ev_share_percent,
                seed=selection_seed,
                salt="ev",
            )
        )
        resolved_heat_pump_ids = set(
            select_households_by_share(
                household_ids=household_ids,
                share_percent=heat_pump_share_percent,
                seed=selection_seed,
                salt="heat_pump",
            )
        )
        resolved_battery_ids = set()
        resolved_pv_ids = set()

    pv_kwp_by_bus = {
        bus_id: float(read_object_value(devices, "pv_kwp"))
        for bus_id, devices in household_devices.items()
        if read_object_value(devices, "pv_kwp", None) is not None
    }

    battery_kwh_by_bus = {
        bus_id: float(read_object_value(devices, "battery_kwh"))
        for bus_id, devices in household_devices.items()
        if read_object_value(devices, "battery_kwh", None) is not None
    }

    resolved_load_scaling_by_bus = {
        bus_id: float(global_load_scaling_factor)
        for bus_id in household_ids
    }

    cleaned_overrides: dict[str, dict[str, Any]] = {}

    for raw_bus_id, override in household_overrides.items():
        bus_id = str(raw_bus_id)

        if bus_id not in household_id_set:
            continue

        cleaned_override: dict[str, Any] = {}

        if "has_ev" in override:
            has_ev = bool(override["has_ev"])
            cleaned_override["has_ev"] = has_ev

            if has_ev:
                resolved_ev_ids.add(bus_id)
            else:
                resolved_ev_ids.discard(bus_id)

        if "has_heat_pump" in override:
            has_heat_pump = bool(override["has_heat_pump"])
            cleaned_override["has_heat_pump"] = has_heat_pump

            if has_heat_pump:
                resolved_heat_pump_ids.add(bus_id)
            else:
                resolved_heat_pump_ids.discard(bus_id)

        if "has_battery" in override:
            has_battery = bool(override["has_battery"])
            cleaned_override["has_battery"] = has_battery

            if has_battery:
                resolved_battery_ids.add(bus_id)
            else:
                resolved_battery_ids.discard(bus_id)

        if "has_pv" in override:
            has_pv = bool(override["has_pv"])
            cleaned_override["has_pv"] = has_pv

            if has_pv:
                resolved_pv_ids.add(bus_id)
            else:
                resolved_pv_ids.discard(bus_id)

        if "load_scaling_factor" in override:
            load_scaling_factor = float(override["load_scaling_factor"])
            cleaned_override["load_scaling_factor"] = load_scaling_factor
            resolved_load_scaling_by_bus[bus_id] = load_scaling_factor

        if cleaned_override:
            cleaned_overrides[bus_id] = cleaned_override

    gridcreator_device_data_by_bus = build_gridcreator_device_data_by_bus(
        household_ids=household_ids,
        household_devices=household_devices,
    )

    load_profile_summary_by_bus = build_load_profile_summary_by_bus(
        network=network,
        household_ids=household_ids,
    )

    ev_availability_summary_by_bus = build_ev_availability_summary_by_bus(
        network=network,
        household_ids=household_ids,
    )

    gridcreator_summary = build_gridcreator_summary(
        household_ids=household_ids,
        household_devices=household_devices,
        load_profile_summary_by_bus=load_profile_summary_by_bus,
        ev_availability_summary_by_bus=ev_availability_summary_by_bus,
    )

    selected_bounds_data = bounds_to_dict(selected_bounds)

    return {
        "configuration_type": "household_scenario_configuration",
        "version": 3,
        "topology_changed": False,
        "description": (
            "Konfiguration von GridCreator-/grid_model-Basisdaten, globalen Zielwerten, "
            "optionalen Lastprofil-Skalierungen und individuellen Anpassungen für "
            "vorhandene Haushalts-/Last-Busse."
        ),
        "network": {
            "network_id": getattr(network, "network_id", None),
            "area_name": getattr(network, "area_name", None),
            "bbox": selected_bounds_data["bbox"],
            "area_km2": selected_bounds_data["area_km2"],
            "household_count": len(household_ids),
            "bus_count": len(getattr(network, "buses", [])),
            "line_count": len(getattr(network, "lines", [])),
            "transformer_count": len(getattr(network, "transformers", [])),
        },
        "gridcreator_source_data": {
            "uses_gridcreator_device_data": uses_gridcreator_defaults,
            "summary": gridcreator_summary,
            "device_data_by_bus": gridcreator_device_data_by_bus,
            "load_profile_summary_by_bus": load_profile_summary_by_bus,
            "ev_availability_summary_by_bus": ev_availability_summary_by_bus,
            "notes": {
                "device_data": (
                    "EV, Wärmepumpe, Batterie und PV werden aus "
                    "network.household_devices übernommen, sofern diese Daten im "
                    "GridNetwork vorhanden sind."
                ),
                "load_profile": (
                    "GridCreator/grid_model liefert originale Lastprofile pro Haushalt. "
                    "Ein Lastprofil ist eine Zeitreihe in kW und beschreibt, wie sich "
                    "der Stromverbrauch eines Haushalts über die Zeit verändert. "
                    "Die Lastprofil-Skalierung ist keine zusätzliche Originalinformation "
                    "aus GridCreator, sondern eine optionale Szenario-Annahme der Map UI."
                ),
                "load_scaling_factor": (
                    "Der technische JSON-Key load_scaling_factor beschreibt die "
                    "Lastprofil-Skalierung. 1.0 bedeutet: originales Lastprofil "
                    "unverändert verwenden. 1.2 bedeutet: alle Werte des Lastprofils "
                    "um 20 Prozent erhöhen. 0.8 bedeutet: alle Werte des Lastprofils "
                    "um 20 Prozent reduzieren."
                ),
            },
        },
        "scenario_assumptions": {
            "uses_gridcreator_defaults": uses_gridcreator_defaults,
            "ev_share_percent": float(ev_share_percent),
            "heat_pump_share_percent": float(heat_pump_share_percent),
            "global_load_scaling_factor": float(global_load_scaling_factor),
            "global_load_scaling_factor_label": "Globale Lastprofil-Skalierung für Szenarien",
            "selection_seed": int(selection_seed),
            "selection_seed_label": "Zufallswert für automatische Zielwert-Verteilung",
            "selection_method": (
                "GridCreator-/grid_model-Gerätedaten werden als Basis verwendet, sofern sie "
                "im GridNetwork vorhanden sind. Globale Zielwerte aus der Map UI "
                "und individuelle Haushalt-Anpassungen werden anschließend als "
                "Overrides angewendet. Wenn keine GridCreator-Gerätedaten vorhanden "
                "sind, werden EV und Wärmepumpe deterministisch über die angegebenen "
                "Szenario-Anteile verteilt; Batterie und PV bleiben ohne Override aus."
            ),
            "load_profile_scaling_method": (
                "Das originale Lastprofil pro Haushalt bleibt bei einer Skalierung von 1.0 "
                "unverändert. Eine globale Lastprofil-Skalierung kann für Szenarien "
                "auf alle aktuell angezeigten Haushalte angewendet werden. Zusätzlich "
                "können einzelne Haushalte eine eigene Skalierung über "
                "individual_household_adjustments erhalten."
            ),
            "note": (
                "Für dieses Netz liegen GridCreator-/grid_model-Gerätedaten vor. Die "
                "endgültige Konfiguration entsteht aus GridCreator-/grid_model-Basisdaten "
                "plus gespeicherten globalen oder individuellen Anpassungen."
                if uses_gridcreator_defaults
                else
                "Für dieses Netz liegen keine GridCreator-Gerätedaten vor. Die "
                "Konfiguration basiert daher auf Szenario-Anteilen und individuellen "
                "Anpassungen."
            ),
        },
        "individual_household_adjustments": cleaned_overrides,
        "resolved": {
            "ev_bus_ids": sorted(resolved_ev_ids),
            "heat_pump_bus_ids": sorted(resolved_heat_pump_ids),
            "battery_bus_ids": sorted(resolved_battery_ids),
            "pv_bus_ids": sorted(resolved_pv_ids),
            "pv_kwp_by_bus": pv_kwp_by_bus,
            "battery_kwh_by_bus": battery_kwh_by_bus,
            "load_scaling_by_bus": resolved_load_scaling_by_bus,
        },
        "resolved_summary": build_resolved_summary(
            household_count=len(household_ids),
            resolved_ev_ids=resolved_ev_ids,
            resolved_heat_pump_ids=resolved_heat_pump_ids,
            resolved_battery_ids=resolved_battery_ids,
            resolved_pv_ids=resolved_pv_ids,
            pv_kwp_by_bus=pv_kwp_by_bus,
            battery_kwh_by_bus=battery_kwh_by_bus,
            load_scaling_by_bus=resolved_load_scaling_by_bus,
        ),
        "assumptions": {
            "ev_means": "Electric Vehicle / Elektrofahrzeug am Haushalt",
            "heat_pump_means": "Wärmepumpe am Haushalt",
            "battery_means": "Batteriespeicher am Haushalt",
            "pv_means": "Photovoltaikanlage am Haushalt",
            "load_scaling_factor_means": (
                "Technischer JSON-Key für die Lastprofil-Skalierung. "
                "1.0 = originales Lastprofil unverändert, "
                "1.2 = Lastprofilwerte um 20 Prozent höher, "
                "0.8 = Lastprofilwerte um 20 Prozent niedriger."
            ),
            "original_load_profile_means": (
                "Das originale Lastprofil stammt aus GridCreator/grid_model und ist "
                "eine Zeitreihe der Haushaltslast in kW."
            ),
            "topology_note": "Die Netztopologie des erzeugten GridNetwork bleibt unverändert.",
            "capacity_note": (
                "PV-Leistung und Batteriekapazität werden im JSON weiterhin als "
                "technische Werte gespeichert, sofern GridCreator/grid_model diese "
                "Werte bereitgestellt hat. Sie müssen aber nicht zwingend in der UI "
                "als Gesamtwerte angezeigt werden."
            ),
        },
    }


def build_gridcreator_device_data_by_bus(
    household_ids: list[str],
    household_devices: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for bus_id in household_ids:
        devices = household_devices.get(bus_id)

        if devices is None:
            result[bus_id] = {
                "has_gridcreator_device_data": False,
                "ev": False,
                "heat_pump": False,
                "battery": False,
                "pv": False,
                "ev_kw": None,
                "heat_pump_kw": None,
                "battery_kwh": None,
                "pv_kwp": None,
            }
            continue

        result[bus_id] = {
            "has_gridcreator_device_data": True,
            "ev": bool(read_object_value(devices, "ev", False)),
            "heat_pump": bool(read_object_value(devices, "heat_pump", False)),
            "battery": bool(read_object_value(devices, "battery", False)),
            "pv": bool(read_object_value(devices, "pv", False)),
            "ev_kw": optional_float(read_object_value(devices, "ev_kw", None)),
            "heat_pump_kw": optional_float(read_object_value(devices, "heat_pump_kw", None)),
            "battery_kwh": optional_float(read_object_value(devices, "battery_kwh", None)),
            "pv_kwp": optional_float(read_object_value(devices, "pv_kwp", None)),
        }

    return result


def build_load_profile_summary_by_bus(
    network: Any,
    household_ids: list[str],
) -> dict[str, dict[str, Any]]:
    profiles = getattr(network, "household_load_profile_kw", {}) or {}
    result: dict[str, dict[str, Any]] = {}

    for bus_id in household_ids:
        profile = profiles.get(bus_id)

        if not profile:
            result[bus_id] = {
                "has_profile": False,
                "steps": 0,
                "timestep_hours": None,
                "daily_energy_kwh": None,
                "peak_kw": None,
                "mean_kw": None,
                "description": "Kein originales Lastprofil vorhanden.",
            }
            continue

        values = [float(value) for value in profile]
        steps = len(values)
        dt_hours = 24.0 / steps if steps else 0.25

        daily_energy_kwh = round(sum(values) * dt_hours, 4)
        peak_kw = round(max(values), 4)
        mean_kw = round(sum(values) / steps, 4) if steps else 0.0

        result[bus_id] = {
            "has_profile": True,
            "steps": steps,
            "timestep_hours": round(dt_hours, 6),
            "daily_energy_kwh": daily_energy_kwh,
            "peak_kw": peak_kw,
            "mean_kw": mean_kw,
            "description": (
                f"Originales Lastprofil mit {steps} Zeitschritten. "
                f"Tagesenergie {daily_energy_kwh} kWh, "
                f"Peak {peak_kw} kW, Durchschnitt {mean_kw} kW."
            ),
        }

    return result


def build_ev_availability_summary_by_bus(
    network: Any,
    household_ids: list[str],
) -> dict[str, dict[str, Any]]:
    availability_by_bus = getattr(network, "ev_availability", {}) or {}
    result: dict[str, dict[str, Any]] = {}

    for bus_id in household_ids:
        availability = availability_by_bus.get(bus_id)

        if not availability:
            result[bus_id] = {
                "has_profile": False,
                "steps": 0,
                "connected_steps": 0,
                "connected_share_percent": None,
            }
            continue

        steps = len(availability)
        connected_steps = sum(1 for value in availability if bool(value))

        result[bus_id] = {
            "has_profile": True,
            "steps": steps,
            "connected_steps": connected_steps,
            "connected_share_percent": (
                round(connected_steps / steps * 100.0, 2) if steps else 0.0
            ),
        }

    return result


def build_gridcreator_summary(
    household_ids: list[str],
    household_devices: dict[str, Any],
    load_profile_summary_by_bus: dict[str, dict[str, Any]],
    ev_availability_summary_by_bus: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    household_count = len(household_ids)

    ev_count = 0
    heat_pump_count = 0
    battery_count = 0
    pv_count = 0
    total_battery_kwh = 0.0
    total_pv_kwp = 0.0

    for bus_id in household_ids:
        devices = household_devices.get(bus_id)

        if devices is None:
            continue

        if bool(read_object_value(devices, "ev", False)):
            ev_count += 1

        if bool(read_object_value(devices, "heat_pump", False)):
            heat_pump_count += 1

        if bool(read_object_value(devices, "battery", False)):
            battery_count += 1
            total_battery_kwh += float(read_object_value(devices, "battery_kwh", 0.0) or 0.0)

        if bool(read_object_value(devices, "pv", False)):
            pv_count += 1
            total_pv_kwp += float(read_object_value(devices, "pv_kwp", 0.0) or 0.0)

    households_with_load_profile = sum(
        1 for summary in load_profile_summary_by_bus.values()
        if summary["has_profile"]
    )
    households_with_ev_availability = sum(
        1 for summary in ev_availability_summary_by_bus.values()
        if summary["has_profile"]
    )

    return {
        "household_count": household_count,
        "households_with_gridcreator_device_data": len(household_devices),
        "households_with_load_profile": households_with_load_profile,
        "households_with_ev_availability": households_with_ev_availability,
        "ev_count": ev_count,
        "heat_pump_count": heat_pump_count,
        "battery_count": battery_count,
        "pv_count": pv_count,
        "ev_share_percent": percent(ev_count, household_count),
        "heat_pump_share_percent": percent(heat_pump_count, household_count),
        "battery_share_percent": percent(battery_count, household_count),
        "pv_share_percent": percent(pv_count, household_count),
        "total_battery_kwh": round(total_battery_kwh, 4),
        "total_pv_kwp": round(total_pv_kwp, 4),
    }


def build_resolved_summary(
    household_count: int,
    resolved_ev_ids: set[str],
    resolved_heat_pump_ids: set[str],
    resolved_battery_ids: set[str],
    resolved_pv_ids: set[str],
    pv_kwp_by_bus: dict[str, float],
    battery_kwh_by_bus: dict[str, float],
    load_scaling_by_bus: dict[str, float],
) -> dict[str, Any]:
    load_factors = list(load_scaling_by_bus.values())

    return {
        "household_count": household_count,
        "ev_count": len(resolved_ev_ids),
        "heat_pump_count": len(resolved_heat_pump_ids),
        "battery_count": len(resolved_battery_ids),
        "pv_count": len(resolved_pv_ids),
        "ev_share_percent": percent(len(resolved_ev_ids), household_count),
        "heat_pump_share_percent": percent(len(resolved_heat_pump_ids), household_count),
        "battery_share_percent": percent(len(resolved_battery_ids), household_count),
        "pv_share_percent": percent(len(resolved_pv_ids), household_count),
        "total_battery_kwh_known": round(sum(float(v) for v in battery_kwh_by_bus.values()), 4),
        "total_pv_kwp_known": round(sum(float(v) for v in pv_kwp_by_bus.values()), 4),
        "load_scaling_min": min(load_factors) if load_factors else None,
        "load_scaling_max": max(load_factors) if load_factors else None,
        "load_scaling_mean": (
            round(sum(load_factors) / len(load_factors), 4)
            if load_factors else None
        ),
        "load_scaling_description": (
            "Diese Werte beschreiben die angewendete Lastprofil-Skalierung. "
            "1.0 bedeutet: originales Lastprofil unverändert."
        ),
    }


def bounds_to_dict(selected_bounds: Any) -> dict[str, Any]:
    if selected_bounds is None:
        return {
            "bbox": None,
            "area_km2": None,
        }

    return {
        "bbox": selected_bounds.model_dump(),
        "area_km2": selected_bounds.approx_area_km2(),
    }


def optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_share_percent(value: Any) -> float:
    """
    Convert a percentage value to a float between 0.0 and 100.0.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0

    return max(0.0, min(100.0, numeric))


def percent(part: int, total: int) -> float:
    if total <= 0:
        return 0.0

    return round(part / total * 100.0, 2)


def read_object_value(obj: Any, attr_name: str, default: Any = None) -> Any:
    """
    Read a value from either a normal object or a dictionary.
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(attr_name, default)

    return getattr(obj, attr_name, default)


def json_dumps_pretty(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)