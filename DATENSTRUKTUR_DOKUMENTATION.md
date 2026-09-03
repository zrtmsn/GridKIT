# GridKIT JSON Datenstruktur Dokumentation

Diese Dokumentation beschreibt die Struktur aller JSON-Dateien die vom GridKIT-Projekt erzeugt werden.

---

## Inhaltsverzeichnis

1. [`summary.json`](#1-summaryjson) — Aggregierte Szenario-Metriken
2. [`timelines.json`](#2-timelinesjson) — Detaillierte Zeitreihen-Daten
3. [`iteration_metrics.json`](#3-iterationmetricsjson) — RLlib Trainings-Metriken

---

## 1. `summary.json`

**Erstellt von:** `src/GridKIT/scripts/run_experiment.py`  
**Verwendet von:** Dashboard (Overview Tab)

### JSON-Struktur

```jsonc
[
  {
    "penetration": <float>,
    "scenario": <string>,
    "curtailment_mean": <float>,
    "curtailment_std": <float>,
    "soc_mean": <float>,
    "soc_std": <float>,
    "peak_mean": <float>,
    "peak_std": <float>,
    "reward_mean": <float>,
    "reward_std": <float>,
    "bill_mean": <float>,
    "bill_std": <float>,
    "hp_comfort_mean": <float>,
    "hp_comfort_std": <float>,
    "battery_charge_kwh_mean": <float>,
    "battery_charge_kwh_std": <float>,
    "battery_discharge_kwh_mean": <float>,
    "battery_discharge_kwh_std": <float>
  }
]
```

### Feld-Erklärungen

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `penetration` | float | EV-Penetration Level (0.2, 0.4, 0.6) |
| `scenario` | string | Szenario-Name |
| `curtailment_mean` | float | Mittelwert: Anzahl Steps mit §14a Netzregelung |
| `curtailment_std` | float | Standardabweichung: Anzahl Steps mit §14a Netzregelung |
| `soc_mean` | float | Mittelwert: State-of-Charge Satisfaction Rate (0-1) |
| `soc_std` | float | Standardabweichung: State-of-Charge Satisfaction Rate |
| `peak_mean` | float | Mittelwert: Transformer Peak Loading (p.u.) |
| `peak_std` | float | Standardabweichung: Transformer Peak Loading |
| `reward_mean` | float | Mittelwert: Mean Episode Reward |
| `reward_std` | float | Standardabweichung: Mean Episode Reward |
| `bill_mean` | float | Mittelwert: Household Electricity Bill (€) |
| `bill_std` | float | Standardabweichung: Household Electricity Bill |
| `hp_comfort_mean` | float | Mittelwert: Heat Pump Comfort Satisfaction (0-1) |
| `hp_comfort_std` | float | Standardabweichung: Heat Pump Comfort Satisfaction |
| `battery_charge_kwh_mean` | float | Mittelwert: Battery Throughput geladen (kWh) |
| `battery_charge_kwh_std` | float | Standardabweichung: Battery Throughput geladen |
| `battery_discharge_kwh_mean` | float | Mittelwert: Battery Throughput entladen (kWh) |
| `battery_discharge_kwh_std` | float | Standardabweichung: Battery Throughput entladen |
| `battery_throughput_kwh_mean` | float | Mittelwert: Gesamte Energie-Bewegung (laden + entladen in kWh) |
| `battery_throughput_kwh_std` | float | Standardabweichung: Gesamte Energie-Bewegung |
| `battery_full_cycles_mean` | float | Mittelwert: Äquivalente Vollzyklen (Full Equivalent Cycles nach IEEE-Standard) |
| `battery_full_cycles_std` | float | Standardabweichung: Äquivalente Vollzyklen |

---

## 2. `timelines.json`

**Erstellt von:** `src/GridKIT/scripts/run_experiment.py`  
**Verwendet von:** Dashboard (Overload Map Tab, Line Charts)

### JSON-Struktur

```jsonc
[
  {
    "penetration": <float>,
    "scenario": <string>,
    "transformer_loading": <list[float]>,
    "max_line_loading": <list[float]>,
    "curtailment": <list[bool]>,
    "overloaded_transformers": <list[list[str]]>,
    "overloaded_lines": <list[list[str]]>,
    "price": <list[float]>,
    "base_load": <list[float]>,
    "pv_generation": <list[float]>,
    "temperature": <list[float]>,
    "ev_power": <list[float]>,
    "battery_power": <list[float]>,
    "hp_power": <list[float]>,
    "house_ev_power": <list[float]>,
    "house_ev_soc": <list[float]>,
    "house_ev_available": <list[float]>,
    "house_battery_power": <list[float]>,
    "house_battery_soc": <list[float]>,
    "house_battery_charge_cumulative_kwh": <list[float]>,
    "house_battery_discharge_cumulative_kwh": <list[float]>,
    "house_hp_power": <list[float]>,
    "house_hp_soc": <list[float]>,
    "house_pv": <list[float]>,
    "soc_satisfaction_rate": <float>
  }
]
```

### Feld-Erklärungen

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `penetration` | float | EV-Penetration Level (0.0–1.0) |
| `scenario` | string | Szenario-Name |
| `transformer_loading` | list[float] | Trafo-Auslastung pro Step (96 Steps, p.u.) |
| `max_line_loading` | list[float] | Maximale Line-Auslastung pro Step (96 Steps, p.u.) |
| `curtailment` | list[bool] | Curtaiment angewendet pro Step (96 Steps) |
| `overloaded_transformers` | list[list[str]] | IDs überlasteter Transformatoren pro Step (96 Steps) |
| `overloaded_lines` | list[list[str]] | IDs überlasteter Lines pro Step (96 Steps) |
| `price` | list[float] | Strompreis pro Step (96 Steps, €/kWh) |
| `base_load` | list[float] | Nicht-steuerbare Last pro Step (96 Steps, kW) |
| `pv_generation` | list[float] | PV-Einspeisung pro Step (96 Steps, kW) |
| `temperature` | list[float] | Außentemperatur pro Step (96 Steps, °C) |
| `ev_power` | list[float] | Alle EVs zusammen pro Step (96 Steps, kW) |
| `battery_power` | list[float] | Alle Batterien zusammen pro Step (96 Steps, kW) |
| `hp_power` | list[float] | Alle Wärmepumpen zusammen pro Step (96 Steps, kW) |
| `house_ev_power` | list[float] | Einzelnes repräsentatives EV pro Step (96 Steps, kW) |
| `house_ev_soc` | list[float] | Einzelnes repräsentatives EV SoC pro Step (96 Steps, 0.0–1.0) |
| `house_ev_available` | list[float] | Einzelnes EV verfügbar pro Step (96 Steps, 0.0–1.0) |
| `house_battery_power` | list[float] | Einzelne repräsentative Batterie pro Step (96 Steps, kW) |
| `house_battery_soc` | list[float] | Einzelne repräsentative Batterie SoC pro Step (96 Steps, 0.0–1.0) |
| `house_battery_charge_cumulative_kwh` | list[float] | Einzelne Batterie: Kumulative geladene Energie (96 Steps, kWh) – für Full Equivalent Cycles |
| `house_battery_discharge_cumulative_kwh` | list[float] | Einzelne Batterie: Kumulative entladene Energie (96 Steps, kWh) – für Full Equivalent Cycles |
| `house_hp_power` | list[float] | Einzelne repräsentative Wärmepumpe pro Step (96 Steps, kW) |
| `house_hp_soc` | list[float] | Einzelne repräsentative HP SoC pro Step (96 Steps, 0.0–1.0) |
| `house_pv` | list[float] | Einzelne repräsentative PV-Anlage pro Step (96 Steps, kW) |
| `soc_satisfaction_rate` | float | Anteil Agenten mit Ziel-SoC (0.0–1.0) |

**Hinweis:** 96 Steps = 24h × 4 Steps/h (15-Minuten-Intervalle)

---

## 3. `iteration_metrics.json`

**Erstellt von:** `src/GridKIT/rl_engine/trainer.py`  
**Verwendet von:** Dashboard (Training Tab)

**Hinweis:** Datei enthält ~200 zusätzliche technische Felder (Config, Timers, Perf-Metrics). Nur für Dashboard relevante Felder sind dokumentiert.

### JSON-Struktur

```jsonc
[
  {
    "training_iteration": <int>,
    "env_runners": {
      "episode_return_mean": <float>,
      "episode_return_min": <float>,
      "episode_return_max": <float>,
      "episode_len_mean": <float>
    },
    "learners": {
      "ev_policy": {
        "policy_loss": <float>,
        "entropy": <float>,
        "vf_loss": <float>,
        "vf_explained_var": <float>
      },
      "battery_policy": {
        "policy_loss": <float>,
        "entropy": <float>,
        "vf_loss": <float>,
        "vf_explained_var": <float>
      },
      "hp_policy": {
        "policy_loss": <float>,
        "entropy": <float>,
        "vf_loss": <float>,
        "vf_explained_var": <float>
      },
    }
  }
]
```

### Feld-Erklärungen

#### Basic Info

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `training_iteration` | int | Iterationsnummer (1, 2, 3, ...) |

#### Episode Rewards (`env_runners.*`)

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `episode_return_mean` | float | Durchschnittlicher Reward aller Episoden dieser Iteration |
| `episode_return_min` | float | Minimaler Reward dieser Iteration |
| `episode_return_max` | float | Maximaler Reward dieser Iteration |
| `episode_len_mean` | float | Durchschnittliche Episodenlänge (Steps) |

#### Policy Metrics (`learners.<policy>.*`)

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `policy_loss` | float | PPO Clip-Loss der Policy |
| `entropy` | float | Policy Entropy (Explorations-Level) |
| `vf_loss` | float | Value Function Loss |
| `vf_explained_var` | float | Varianz der Returns erklärt durch Value Function (höher = besser) |

**Hinweis:** `learners` enthält gleiche Struktur für jede Policy: `ev_policy`, `battery_policy`, `hp_policy`

---

## Datei-Pfade im Output-Verzeichnis

```
outputs/{experiment_name}/
├── summary.json              # Aggregierte Metriken
├── timelines.json            # Zeitreihen-Daten
├── graphs/                   # PNGs von plot_results
└── checkpoints/
    ├── pen_20/
    │   └── iteration_metrics.json
    ├── pen_40/
    │   └── ...
    └── pen_60/
        └── ...
```

---

## Verwendung im Dashboard

| Tab | Verwendete Dateien | Visualisierung |
|-----|-------------------|----------------|
| 🗺️ Overload Map | `timelines.json` + Netzwerk-JSON | Folium Karte |
| 📊 Metrics (PNGs) | `graphs/*.png` | Statische PNG-Bilder |
| 🎯 Training (Rewards) | `checkpoints/*/iteration_metrics.json` | Plotly Charts |

---

*Letzte Aktualisierung: 30. August 2026*