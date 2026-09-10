#!/usr/bin/env python3
"""
Data Loader für Dashboard Prototype.

Lädt timelines.json und extrahiert Overload-Daten für die Karten-Visualisierung.
"""

import json
from pathlib import Path
from typing import Any


def load_timelines(results_dir: str | Path = "outputs_test_weak") -> list[dict]:
    """Lade timelines.json aus dem angegebenen Output-Verzeichnis."""
    timelines_path = Path(results_dir) / "timelines.json"
    
    if not timelines_path.exists():
        raise FileNotFoundError(
            f"❌ FEHLER: '{timelines_path}' existiert nicht!\n"
            f"Zuerst Training laufen lassen:\n"
            f"  python -m GridKIT.scripts.run_experiment \\\n"
            f"      --network data/stub_network_weak_branches.json \\\n"
            f"      --out {results_dir} \\\n"
            f"      --iterations 1 --seeds 1"
        )
    
    with open(timelines_path) as f:
        return json.load(f)


def extract_overloaded_lines(timelines: list[dict]) -> dict[str, dict]:
    """
    Extrahiere überlastete Lines aus allen Szenarien.
    
    Rückgabe: Dict mit Szenario-Namen als Keys und Overload-Daten als Values.
    """
    result = {}
    
    for entry in timelines:
        scenario_key = f"{entry['scenario']} @ {int(entry['penetration'] * 100)}%"
        
        # Alle Steps durchgehen
        overload_by_step = {}
        all_lines = set()
        
        for step_idx, step_lines in enumerate(entry.get('overloaded_lines', [])):
            if step_lines:  # Nicht leer
                lines_set = set(step_lines)
                overload_by_step[step_idx] = lines_set
                all_lines.update(lines_set)
        
        result[scenario_key] = {
            'entry': entry,
            'overload_by_step': overload_by_step,
            'all_lines': all_lines,
            'total_steps': len(overload_by_step),
        }
    
    return result


def get_line_coordinates(network_path: str | Path = "data/stub_network.json") -> dict[str, tuple]:
    """
    Lade Line-Koordinaten aus Netzwerk-JSON.
    
    Rückgabe: Dict {line_id: (from_coords, to_coords)}
              where coords = (lat, lon) = (y_coord, x_coord)
    """
    network_file = Path(network_path)
    if not network_file.exists():
        return {}
    
    with open(network_file) as f:
        network = json.load(f)
    
    # Bus-Koordinaten mappen
    bus_coords = {}
    for bus in network.get('buses', []):
        bus_id = bus['bus_id']
        # Folium erwartet [lat, lon] = [y, x]
        bus_coords[bus_id] = (bus.get('y_coord', 0), bus.get('x_coord', 0))
    
    # Line-Koordinaten extrahieren
    line_coords = {}
    for line in network.get('lines', []):
        line_id = line['line_id']
        from_bus = line['from_bus']
        to_bus = line['to_bus']
        
        if from_bus in bus_coords and to_bus in bus_coords:
            line_coords[line_id] = {
                'from': bus_coords[from_bus],
                'to': bus_coords[to_bus],
            }
    
    return line_coords


def get_household_coordinates(network_path: str | Path = "data/stub_network.json") -> list[dict]:
    """
    Lade Haushalts-Koordinaten aus Netzwerk-JSON.
    
    Rückgabe: Liste von Dicts mit bus_id und coords.
    """
    network_file = Path(network_path)
    if not network_file.exists():
        return []
    
    with open(network_file) as f:
        network = json.load(f)
    
    household_ids = set(network.get('household_bus_ids', []))
    
    households = []
    for bus in network.get('buses', []):
        if bus['bus_id'] in household_ids:
            households.append({
                'bus_id': bus['bus_id'],
                'coords': (bus.get('y_coord', 0), bus.get('x_coord', 0)),
            })
    
    return households


def get_transformer_coordinates(network_path: str | Path = "data/stub_network.json") -> list[dict]:
    """
    Lade Transformator-Koordinaten aus Netzwerk-JSON.
    
    Rückgabe: Liste von Dicts mit trafo_id, lv_bus und coords.
    """
    network_file = Path(network_path)
    if not network_file.exists():
        return []
    
    with open(network_file) as f:
        network = json.load(f)
    
    transformers = []
    for trafo in network.get('transformers', []):
        # LV-Bus Koordinaten finden
        lv_bus_id = trafo.get('lv_bus')
        
        if lv_bus_id is None:
            continue
        
        # Bus-Koordinaten aus buses-Liste holen
        for bus in network.get('buses', []):
            if bus['bus_id'] == lv_bus_id:
                transformers.append({
                    'trafo_id': trafo.get('trafo_id', lv_bus_id),
                    'lv_bus': lv_bus_id,  # ← Wichtig für Match!
                    'coords': (bus.get('y_coord', 0), bus.get('x_coord', 0)),
                })
                break
    
    return transformers