# GridKIT Daten-Pipeline Dokumentation

Diese Dokumentation erklärt den kompletten Datenfluss von der Netzwerkerstellung über das Training bis zur Visualisierung.

---

## 🧰 Voraussetzungen (einmalig)

Alle Befehle in dieser Dokumentation werden **im Repo-Wurzelverzeichnis** ausgeführt — dem
Ordner, in dem `src/`, `pyproject.toml` und `.venv/` liegen. Der exakte Installationsort ist
dabei egal (z. B. `cd "$(git rev-parse --show-toplevel)"`), nur die **relativen Pfade**
(`src/...`, `data/...`, `.venv/...`) bleiben überall gleich.

Umgebung einmal pro Terminal-Sitzung aktivieren:

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)/src/GridKIT:$(pwd)/src"   # GridKIT ist nicht pip-installiert → Quelle sind die src-Ordner
```

Ab dann funktionieren alle `python -m GridKIT.scripts.*`- und `python src/GridKIT/...`-Befehle
aus dieser Dokumentation unverändert. Wer die Package-Installation nutzt (`pip install -e .`),
braucht den `PYTHONPATH`-Export nicht.

---


## 📊 Übersicht: Die 4 Phasen der Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  1. NETWORK     │ →   │  2. TRAINING     │ →   │  3. RESULTS      │ →   │  4. VISUALIZATION│
│  BUILDING       │     │  (Experiment)    │     │  STORAGE         │     │  (Plots)         │
└─────────────────┘     └──────────────────┘     └───────────────────┘     └──────────────────┘
      │                        │                         │                        │
      ▼                        ▼                         ▼                        ▼
  grid_network.json       outputs/                 summary.json            *.png Bilder
  household_configuration.json                timelines.json            (Dashboard-ready)
                                              checkpoints/
```

---

## Phase 1: Netzwerk-Erstellung 🗺️

### 1.1 Map UI (Streamlit)

**Befehl:**
```bash
streamlit run src/GridKIT/map_ui/map_widget.py
```

**Was passiert:**
1. Benutzer wählt Bereich auf Karte (OSM Overpass API)
2. GridCreator wird via `conda run` ausgeführt
3. Ergebnis: `GridNetwork` Pydantic-Modell

**Ausgabedateien (Download-Buttons in UI):**
- `grid_network.json` – Netzwerk-Topologie + Haushalts-Konfiguration
- `household_configuration.json` – EV/WP-Verteilung pro Haushalt

**Speicherort der Daten im Code:**
```python
# src/GridKIT/map_ui/map_widget.py Zeile 223-226
st.session_state["built_network"] = network
st.session_state["built_bounds"] = selected_bounds
st.session_state["built_scenario"] = scenario
```

---

### 1.2 CLI (Stub oder ding0-Archiv)

**Befehle:**
```bash
# Stub-Netzwerk (5 Haushalte, Testzwecke)
python -m grid_model.cli build --stub --out network.json

# ding0-Archiv (reale deutsche Netze)
python -m grid_model.cli build --ding0 \
    --south 49.0 --west 8.0 --north 49.1 --east 8.1 \
    --out network.json
```

**Generierte Struktur (`network.json`):**
```json
{
  "network_id": "stub_5_households",
  "buses": [...],
  "lines": [...],
  "transformers": [...],
  "household_bus_ids": ["bus_1", "bus_3", ...],
  "household_load_profile_kw": {"bus_1": [0.5, 0.6, ...]},
  "ev_availability": {"bus_1": [false, true, ...]}
}
```

---

## Phase 2: Training (Experiment) 🎯

### 2.1 Start-Command <sub>(ausgeführt vom Repo-Wurzelverzeichnis — siehe Voraussetzungen oben)</sub>

**Befehl:**
```bash
export PYTHONPATH="$(pwd)/src/GridKIT:$(pwd)/src"
python -m GridKIT.scripts.run_experiment \
    --network data/stub_network.json \
    --out outputs \
    --max-iterations 50 \
    --seeds 12
```

**Parameter:**
- `--max-iterations`: **Obergrenze (Ceiling)** der IPPO-Training-Iterationen pro Penetration-Level (default: 50, aus `PIPELINE_MAX_TRAINING_ITERATIONS`). Das Training stoppt **vorher automatisch**, sobald der mittlere Episoden-Reward konvergiert ist (Reward-Plateau) — die Netzwerkgröße bestimmt damit die tatsächliche Trainingsdauer, nicht eine fixe Zahl.
- `--min-iterations`: Frühester möglicher Konvergenz-Stopp (default: 5) — schützt vor einem Abbruch auf einem zufälligen Anfangswert.
- `--patience`: Abbruch nach N Iterationen ohne Verbesserung des Rewards (default: 6).
- `smooth_window` (kein separater CLI-Flag; Setting `PIPELINE_EARLY_STOP_SMOOTH_WINDOW`, default: 4): gleitendes Fenster `k` für den **geglätteten** Reward. Der Plateau-Vergleich arbeitet auf dem Durchschnitt der *letzten k* Iterations-Reward-Mittelwerte (das Fenster verschiebt sich — es ist **kein** Durchschnitt aller Werte seit Trainingsbeginn und **kein** erneutes Mitteln bereits gemittelter Werte). So zählt ein einzelner Glücks-Rauschpeak nicht mehr als Fortschritt, und der Stopppunkt hängt nicht mehr von der Position des Rauschmaximums ab. `k=1` deaktiviert die Glättung (unglättet, wie früher).
- `--seeds`: Evaluierungs-Episoden pro Szenario (default: 12)
- `--network`: Pfad zur `grid_network.json` aus Phase 1
- `--out`: Ausgabeverzeichnis (default: `outputs/`)

**Hinweis:** Die Batch-Größen (`train_batch_size`/`minibatch_size` und die Anzahl der Env-Runner) werden automatisch mit der Agentenzahl des Netzwerks skaliert (Referenz: der 6-Agenten-Minimal-Stub), sodass die Transitions **pro Agent pro PPO-Update** bei jeder Netzwerkgröße konstant bleiben — dadurch gilt „Konvergenz nach ~N Updates" unabhängig von der vom Nutzer konfigurierten Netzwerkgröße (Details: `rl_engine/convergence.py`, `rl_engine/trainer.py`).

---

### 2.2 Was im Training passiert

**Schritt-für-Schritt-Ablauf:**

1. **Für jeden EV-Penetrations-Level** (20%, 40%, 60%):
   ```python
   for pen in [0.2, 0.4, 0.6]:
       # 1. IPPO Policy trainieren (Scenario 3: selfish RL)
       #    max_iterations ist die OBERE Grenze: der Trainer stoppt früher,
       #    sobald der Reward auf einem Plateau ist (Early Stop, s. convergence.py).
       trainer.run(max_iterations=args.max_iterations,
                   min_iterations=args.min_iterations, patience=args.patience)
       
       # 2. Checkpoint speichern
       trainer.save_checkpoint(f"outputs/checkpoints/pen_{int(pen*100)}")
       
       # 3. Evaluation gegen 3 Baseline-Szenarien
       scenarios = {
           "1: flat / immediate": NaiveImmediatePolicy(),
           "2: price-follow (manual)": NaivePriceFollowPolicy(jitter_std=8.0),
           "2: price-follow (automated)": NaivePriceFollowPolicy(jitter_std=0.0),
           "3: selfish RL": adapter,  # Trainierte Policy
       }
   ```

2. **Pro Szenario werden gespeichert:**
   - **Metriken**: Curtaiment-Rate, SoC-Zufriedenheit, Peak-Loading, Reward, Kosten
   - **Timelines**: Zeitreihen einer repräsentativen Episode (96 Steps = 24h)

---

### 2.3 Geloggte Daten (pro Episode)

**Aus `scenarios/runner.py`:**

```python
@dataclass
class EpisodeMetrics:
    episode: int
    mean_episode_reward: float
    soc_satisfaction_rate: float       # % EVs mit Ziel-SoC erreicht
    curtailment_events: int            # Anzahl §14a Eingriffe
    transformer_peak_loading_pu: float # Maximaler Trafo-Load (p.u.)
    mean_household_bill_eur: float     # Durchschnittliche Kosten/Haushalt
    hp_comfort_satisfaction_rate: float
    battery_charge_kwh: float
    battery_discharge_kwh: float
    
    # Detaillierte Netz-Analyse:
    feeder_overload_steps: dict[str, int]      # {feeder_id: count}
    feeder_peak_loading_pu: dict[str, float]   # {feeder_id: peak_pu}
    line_overload_steps: dict[str, int]        # {line_id: count}
    line_peak_loading_pu: dict[str, float]     # {line_id: peak_pu}
```

**Zeitreihen-Daten (pro Timestep, 96 Steps):**
```python
{
    "transformer_loading_pu": [0.8, 0.85, 0.92, ...],
    "max_line_loading": [0.6, 0.65, 0.71, ...],
    "curtailment": [False, False, True, ...],
    "price": [0.28, 0.27, 0.29, ...],
    "base_load": [2.5, 2.8, 3.1, ...],
    "pv_generation": [0, 0, 0.5, 2.1, ...],
    "ev_power": [0, 5.2, 10.4, ...],
    "battery_power": [0, -2.1, 3.5, ...],
    "hp_power": [0, 3.0, 3.0, ...],
    "house_ev_soc": [0.55, 0.65, 0.75, ...],
}
```

---

### 2.4 Ausgabedateien des Trainings

**Verzeichnisstruktur:**
```
outputs/
├── summary.json          # Aggregierte Metriken (mean±std)
├── timelines.json        # Zeitreihen repräsentativer Episoden
└── checkpoints/
    ├── pen_20/           # IPPO Checkpoint für 20% EV-Penetration
    ├── pen_40/
    └── pen_60/
```

**`summary.json` Format:**
```json
[
  {
    "penetration": 0.2,
    "scenario": "1: flat / immediate",
    "curtailment_mean": 5.2,
    "curtailment_std": 1.3,
    "soc_mean": 0.95,
    "peak_mean": 1.15,
    "reward_mean": -12.5,
    "bill_mean": 4.20
  }
]
```

**`timelines.json` Format:**
```json
[
  {
    "penetration": 0.2,
    "scenario": "3: selfish RL",
    "transformer_loading": [...],
    "ev_power": [...],
    "house_ev_soc": [...]
  }
]

---

## Phase 3: Daten für Dashboard 📁

### 3.1 Daten laden

```python
import json
from pathlib import Path

results_dir = Path("outputs")
summary = json.loads((results_dir / "summary.json").read_text())
timelines = json.loads((results_dir / "timelines.json").read_text())
```

### 3.2 Verfügbare Metriken

| Metrik | Beschreibung | Quelle |
|--------|--------------|--------|
| `curtailment_mean` | §14a Eingriffe | summary.json |
| `soc_mean` | EV SoC-Zufriedenheit | summary.json |
| `peak_mean` | Trafo-Load (p.u.) | summary.json |
| `bill_mean` | Kosten/Haushalt [€] | summary.json |
| `transformer_loading` | Zeitreihe | timelines.json |
| `ev_power` | Ladeleistung [kW] | timelines.json |

---

### 3.3 Overload-Logging für Dashboard-Visualisierung 🚨

**Seit: August 2026 Fix in `environment.py`**

#### Problem vor dem Fix:
```python
# VORHER: Inkonsistentes Logging ❌
curtailment_applied = True      # §14a wurde ausgelöst!
overloaded_lines = []           # Aber keine Line überlastet?!
```

**Grund:** Es wurden die Werte NACH erfolgreichem Curtailment geloggt (< 1.0 p.u.), nicht die Werte die das Curtailment AUSLÖSTEN (> 1.0 p.u.).

#### Lösung: Loggen der VOR-Curtailment Werte ✅

**Code-Änderung in `src/GridKIT/grid_model/environment.py` (Zeile 286-323):**
```python
# 2. Power flow BEFORE curtailment
max_loading_before, lines_before, volts = self._solve(self._household_nets(req))
feeder_loadings_before = dict(self._surrogate.last_feeder_loadings)

# 3. Prüfen ob Overload vorliegt
overloaded = (any(v > thresh for v in feeder_loadings_before.values())
              or any(l > const.LINE_OVERLOAD_THRESHOLD for l in lines_before.values()))

if overloaded:
    # ... Curtailment wird angewendet ...
    
    # CONSISTENCY FIX: Log BEFORE values (the overload that triggered §14a)
    max_loading = max_loading_before
    feeder_loadings = feeder_loadings_before
    lines = lines_before
else:
    # No curtailment → use original values
    max_loading = max_loading_before
    feeder_loadings = feeder_loadings_before
    lines = lines_before
```

#### Ergebnis nach dem Fix:
```python
# NACHHER: Konsistentes Logging ✅
curtailment_applied = True      # §14a wurde ausgelöst!
overloaded_lines = ['lv_to_0']  # Line war bei 1.05 p.u. (> 1.0!)
```

#### Datenstruktur in `timelines.json`:

**Pro Szenario-Eintrag:**
```json
{
  "penetration": 0.6,
  "scenario": "2: price-follow (manual)",
  "curtailment": [false, false, true, ...],
  "overloaded_lines": [
    [],                    // Step 0: keine Überlastung
    [],                    // Step 1: keine Überlastung
    ["lv_to_0", "1_to_2"], // Step 2: BEIDE Lines > 1.0 p.u.!
    ["1_to_2"],            // Step 3: nur diese Line
    ...
  ],
  "overloaded_transformers": [
    [],           // Step 0: kein Transformer overloading
    ["trafo_0"],  // Step 1: Transformer bei 1.05 p.u.
    ...
  ]
}
```

#### Konsistenz-Check:

**Test-Skript:** `src/GridKIT/scripts/test_overload_logging.py`

Das Skript **startet kein Training** — es analysiert nur die `timelines.json` aus einem
vorangegangenen `run_experiment`-Lauf. Soll die Datei noch fehlen, erst Daten erzeugen
(siehe „So testest du die Overload-Logging-Konsistenz" weiter unten).

```bash
# Vom Repo-Wurzelverzeichnis (siehe „Voraussetzungen" oben):
python src/GridKIT/scripts/test_overload_logging.py outputs_test_weak
```

**Erwartete Ausgabe (Auszug):**
```
======================================================================
OVERLOAD-LOGGING ANALYSIS
...
1: flat / immediate @ 20%:
  Curtailment Steps: 3
  Overload Steps: 3
  Unique overloaded lines: 2
  ✅ CONSISTENT: Curtailment = Overload Steps
...
======================================================================
SUMMARY
...
✅ ALL CONSISTENT: Curtailment always paired with overloads!
```

#### Verwendung im Dashboard:

**Beispiel: Overloaded Lines auf Karte markieren**
```python
import json
from pathlib import Path

timelines = json.load(open("outputs/timelines.json"))

for entry in timelines:
    scenario = entry["scenario"]
    
    # Alle überlasteten Lines extrahieren
    all_overloaded_lines = set()
    for step_lines in entry["overloaded_lines"]:
        all_overloaded_lines.update(step_lines)
    
    print(f"{scenario}: {len(all_overloaded_lines)} unique lines overloaded")
    
    # Auf Folium-Karte rot markieren
    for line_id in all_overloaded_lines:
        highlight_line_on_map(line_id, color="red")
```

#### Test-Netzwerk für mehr Overloads:

Für aussagekräftigere Tests wurde ein Netzwerk mit schwachen Zweigleitungen erstellt:

**Datei:** `data/stub_network_weak_branches.json`

```json
{
  "network_id": "stub_5_households_weak_branches",
  "lines": [
    {"line_id": "lv_to_0", "max_i_ka": 0.060},  // Stark (Hauptleitung)
    {"line_id": "0_to_1", "max_i_ka": 0.025},   // Schwach
    {"line_id": "1_to_2", "max_i_ka": 0.020},   // Sehr schwach
    {"line_id": "2_to_3", "max_i_ka": 0.015},   // Extrem schwach
    {"line_id": "3_to_4", "max_i_ka": 0.010}    // Am schwächsten
  ]
}
```

**So testest du die Overload-Logging-Konsistenz (Schritt für Schritt):**

**Schritt 1 — Trainingsdaten erzeugen** (vom Repo-Wurzelverzeichnis; erzeugt
`outputs_test_weak/{summary.json, timelines.json, checkpoints/}`):

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)/src/GridKIT:$(pwd)/src"
python -m GridKIT.scripts.run_experiment \
    --network data/stub_network_weak_branches.json \
    --out outputs_test_weak \
    --max-iterations 1 --seeds 1
```

(≈ 40 s mit 1 Iteration × 3 Penetrationsstufen; die Defaults von `--max-iterations`
kommen aus `PIPELINE_MAX_TRAINING_ITERATIONS`, das Training stoppt dank
Konvergenz-Erkennung ohnehin früher.)

**Schritt 2 — Konsistenz analysieren** (kein Training, nur Auswertung):

```bash
python src/GridKIT/scripts/test_overload_logging.py outputs_test_weak
```

**Ergebnis:** Bis zu **5 verschiedene Lines** gleichzeitig überlastet bei 60% Penetration!

---

## Phase 4: Visualisierung 📊

### 4.1 Plot-Script

```bash
python -m GridKIT.scripts.plot_results \
    --results outputs --out outputs/graphs
```

### 4.2 Generierte PNGs

| Datei | Inhalt |
|-------|--------|
| `curtailment_events.png` | §14a Eingriffe |
| `soc_satisfaction.png` | EV SoC-Rate |
| `household_bill.png` | Stromkosten |
| `peak_loading.png` | Trafo-Last |
| `episode_timeline_*.png` | Zeitreihen |

---

## Quick-Look ohne Training ⚡

```bash
python -m grid_model.cli plot network.json --out-dir outputs/quicklook
```

Generiert 4 PNGs mit Heuristic-Policy.

---

## Datenfluss-Übersicht

```
Map UI/CLI → grid_network.json
     ↓
run_experiment.py → summary.json + timelines.json
     ↓
plot_results.py → outputs/graphs/*.png
     ↓
Dashboard → Interaktive Visualisierung
```

---

## Wichtige Konstanten

| Konstante | Wert | Bedeutung |
|-----------|------|-----------|
| `EPISODE_STEPS` | 96 | 24h / 15min |
| `EV_TARGET_SOC` | 0.80 | Ziel-Ladezustand |
| `TRANSFORMER_OVERLOAD_THRESHOLD` | 1.0 | p.u. Limit |

---

**Erstellt:** 30.08.2026 | **Getestet:** Stub-Network ✅

```