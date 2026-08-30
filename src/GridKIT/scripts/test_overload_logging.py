#!/usr/bin/env python3
"""
Test-Skript zur Überprüfung der Overload-Logging-Konsistenz.

Analysiert timelines.json und prüft:
1. Curtailment Steps == Overload Steps (Konsistenz)
2. Welche Lines waren in welchen Steps überlastet?
3. Wie oft war jede Line betroffen?

Verwendung:
    python test_overload_logging.py [output_dir]
    
Beispiel:
    python test_overload_logging.py outputs_test_weak
"""

import json
import sys
from pathlib import Path


def analyze_overloads(timelines_path: Path) -> None:
    """Analysiere Overload-Daten aus timelines.json."""
    
    if not timelines_path.exists():
        print(f"❌ FEHLER: '{timelines_path}' existiert nicht!")
        print()
        print("Zuerst Training/Test laufen lassen:")
        print("  python -m GridKIT.scripts.run_experiment \\")
        print("      --network data/stub_network.json \\")
        print("      --out outputs_test \\")
        print("      --iterations 1 --seeds 1")
        print()
        print("Oder mit schwachen Zweigen (mehr Overloads):")
        print("  python -m GridKIT.scripts.run_experiment \\")
        print("      --network data/stub_network_weak_branches.json \\")
        print("      --out outputs_test_weak \\")
        print("      --iterations 1 --seeds 1")
        sys.exit(1)
    
    with open(timelines_path) as f:
        data = json.load(f)
    
    print("=" * 70)
    print("OVERLOAD-LOGGING ANALYSE")
    print("=" * 70)
    print(f"Quelle: {timelines_path}")
    print(f"Szenarien: {len(data)}")
    print()
    
    total_scenarios = len(data)
    scenarios_with_curtailment = 0
    scenarios_with_overloads = 0
    inconsistencies = []
    
    for entry in data:
        scenario_name = entry.get('scenario', 'Unknown')
        penetration = entry.get('penetration', 0)
        
        curtail_steps = sum(1 for x in entry.get('curtailment', []) if x)
        
        # Robust: Felder könnten fehlen (alte timelines.json vor dem Fix)
        overloaded_lines = entry.get('overloaded_lines', [])
        overloaded_transformers = entry.get('overloaded_transformers', [])
        
        overload_steps = []
        for i, step_lines in enumerate(overloaded_lines):
            if step_lines:
                overload_steps.append((i, set(step_lines)))
        
        transformer_steps = []
        for i, step_trafos in enumerate(overloaded_transformers):
            if step_trafos:
                transformer_steps.append((i, set(step_trafos)))
        
        has_curtailment = curtail_steps > 0
        has_overloads = len(overload_steps) > 0 or len(transformer_steps) > 0
        
        if has_curtailment:
            scenarios_with_curtailment += 1
        if has_overloads:
            scenarios_with_overloads += 1
        
        if has_curtailment and not has_overloads:
            inconsistencies.append({
                'scenario': scenario_name,
                'penetration': penetration,
                'curtail_steps': curtail_steps,
                'issue': 'Curtailment aber KEINE Overloads!'
            })
        elif has_overloads and not has_curtailment:
            inconsistencies.append({
                'scenario': scenario_name,
                'penetration': penetration,
                'overload_steps': len(overload_steps),
                'issue': 'Overloads aber KEIN Curtailment!'
            })
        
        if curtail_steps > 0:
            print(f"{scenario_name} @ {penetration*100:.0f}%:")
            print(f'  Curtailment Steps: {curtail_steps}')
            print(f'  Overload Steps: {len(overload_steps)}')
            
            if transformer_steps:
                print(f'  Transformer Overload Steps: {len(transformer_steps)}')
            
            line_counts = {}
            for _, lines in overload_steps:
                for line in lines:
                    line_counts[line] = line_counts.get(line, 0) + 1
            
            trafo_counts = {}
            for _, trafos in transformer_steps:
                for trafo in trafos:
                    trafo_counts[trafo] = trafo_counts.get(trafo, 0) + 1
            
            if line_counts:
                print(f'  Einzigartige überlastete Lines: {len(line_counts)}')
                for line, count in sorted(line_counts.items(), key=lambda x: -x[1]):
                    print(f'    {line}: {count} Steps')
            
            if trafo_counts:
                print(f'  Einzigartige überlastete Transformers: {len(trafo_counts)}')
                for trafo, count in sorted(trafo_counts.items(), key=lambda x: -x[1]):
                    print(f'    {trafo}: {count} Steps')
            
            total_overload_steps = len(overload_steps) + len(transformer_steps)
            if curtail_steps == total_overload_steps:
                print(f'  ✅ KONSISTENT: Curtailment = Overload Steps')
            else:
                print(f'  ⚠️  ABWEICHUNG: Curtailment ({curtail_steps}) ≠ Overload ({total_overload_steps})')
            
            if len(overload_steps) <= 10:
                print(f'  Details:')
                for step, lines in overload_steps:
                    print(f'    Step {step:2d}: {lines}')
            else:
                print(f'  Erste 5 Steps:')
                for step, lines in overload_steps[:5]:
                    print(f'    Step {step:2d}: {lines}')
            print()
    
    print("=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)
    print(f"Szenarien insgesamt: {total_scenarios}")
    print(f"Szenarien mit Curtailment: {scenarios_with_curtailment}")
    print(f"Szenarien mit Overloads: {scenarios_with_overloads}")
    print()
    
    if inconsistencies:
        print(f"❌ INKONSISTENZEN GEFUNDEN: {len(inconsistencies)}")
        for inc in inconsistencies:
            print(f"  - {inc['scenario']} @ {inc['penetration']*100:.0f}%: {inc['issue']}")
    else:
        print("✅ ALLE KONSISTENT: Curtailment immer mit Overloads gepaart!")
    
    print()


def main():
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("outputs_test_weak")
    
    timelines_path = output_dir / "timelines.json"
    analyze_overloads(timelines_path)


if __name__ == "__main__":
    main()
