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


DEFAULT_EV_SHARE_PERCENT = 30
DEFAULT_HEAT_PUMP_SHARE_PERCENT = 20
DEFAULT_GLOBAL_LOAD_SCALING_FACTOR = 1.0
DEFAULT_SELECTION_SEED = 42


def default_scenario_assumptions() -> dict[str, Any]:
    return {
        "ev_share_percent": DEFAULT_EV_SHARE_PERCENT,
        "heat_pump_share_percent": DEFAULT_HEAT_PUMP_SHARE_PERCENT,
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
) -> dict[str, Any]:
    scenario_ev_ids = set(
        select_households_by_share(
            household_ids=household_ids,
            share_percent=ev_share_percent,
            seed=selection_seed,
            salt="ev",
        )
    )

    scenario_heat_pump_ids = set(
        select_households_by_share(
            household_ids=household_ids,
            share_percent=heat_pump_share_percent,
            seed=selection_seed,
            salt="heat_pump",
        )
    )

    resolved_ev_ids = set(scenario_ev_ids)
    resolved_heat_pump_ids = set(scenario_heat_pump_ids)

    resolved_load_scaling_by_bus = {
        bus_id: float(global_load_scaling_factor)
        for bus_id in household_ids
    }

    cleaned_overrides: dict[str, dict[str, Any]] = {}

    for bus_id, override in household_overrides.items():
        if bus_id not in household_ids:
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

        if "load_scaling_factor" in override:
            load_scaling_factor = float(override["load_scaling_factor"])
            cleaned_override["load_scaling_factor"] = load_scaling_factor
            resolved_load_scaling_by_bus[bus_id] = load_scaling_factor

        if cleaned_override:
            cleaned_overrides[bus_id] = cleaned_override

    return {
        "configuration_type": "household_scenario_configuration",
        "version": 1,
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
            "heat_pump_share_percent": int(heat_pump_share_percent),
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
        "resolved": {
            "ev_bus_ids": sorted(resolved_ev_ids),
            "heat_pump_bus_ids": sorted(resolved_heat_pump_ids),
            "load_scaling_by_bus": resolved_load_scaling_by_bus,
        },
        "assumptions": {
            "ev_means": "Electric Vehicle / Elektrofahrzeug am Haushalt",
            "heat_pump_means": "Wärmepumpe am Haushalt",
            "load_scaling_factor_means": (
                "1.0 = unverändertes Lastprofil, 1.2 = 20 Prozent höherer Verbrauch"
            ),
            "topology_note": "Die Netztopologie des erzeugten GridNetwork bleibt unverändert.",
        },
    }


def json_dumps_pretty(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)