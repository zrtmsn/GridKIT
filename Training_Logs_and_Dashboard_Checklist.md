# Training JSON Data Structure Documentation / "Dashboard-Checklist"

This documentation describes the structure of the three JSON files produced during training to be consumed by the Dashboard.

Each table shows which attributes the Dashboard consumes at the state of GridKIT's initial release. This information is useful for understanding which fields the team agreed upon for the Dashboard and what else could be utilised for further development.

Note: During development, this was refered to as "Dashboard-Checklist".

---

## Table of Contents

1. [`summary.json`](#1-summaryjson) — Aggregated scenario metrics
2. [`timelines.json`](#2-timelinesjson) — Detailed time-series data
3. [`iteration_metrics.json`](#3-iterationmetricsjson) — RLlib training metrics

---

## 1. `summary.json`

**Created by:** `src/GridKIT/scripts/run_experiment.py` (batch sweep) and `src/GridKIT/scripts/train_run.py` (map UI runs, via `core.run_store.save_results()`)
**Used by:** Dashboard (Overview tab)

### JSON Structure

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

### Field Descriptions

| Field | Type | Description | Currently consumed by Dashboard |
|------|-----|--------------|-------------------------------|
| `penetration` | float | EV penetration level (0.2, 0.4, 0.6) | Yes – all tabs (scenario × penetration selection) |
| `scenario` | string | Scenario name | Yes – all tabs |
| `curtailment_mean` | float | Mean: number of steps with §14a grid control | Yes – Overview, Scenario comparison |
| `curtailment_std` | float | Std. deviation: number of steps with §14a grid control | Yes – Scenario comparison (spread) |
| `soc_mean` | float | Mean: state-of-charge satisfaction rate (0-1) | Yes – Scenario comparison, Devices |
| `soc_std` | float | Std. deviation: state-of-charge satisfaction rate | Yes – Scenario comparison (spread) |
| `peak_mean` | float | Mean: transformer peak loading (p.u.) | Yes – Overview |
| `peak_std` | float | Std. deviation: transformer peak loading | No |
| `reward_mean` | float | Mean: mean episode reward | Yes – Training (strategy comparison) |
| `reward_std` | float | Std. deviation: mean episode reward | Yes – Training (strategy comparison spread) |
| `bill_mean` | float | Mean: household electricity bill (€) | Yes – Scenario comparison |
| `bill_std` | float | Std. deviation: household electricity bill | Yes – Scenario comparison (spread) |
| `hp_comfort_mean` | float | Mean: heat pump comfort satisfaction (0-1) | Yes – Scenario comparison, Devices |
| `hp_comfort_std` | float | Std. deviation: heat pump comfort satisfaction | Yes – Scenario comparison (spread) |
| `battery_charge_kwh_mean` | float | Mean: battery throughput charged (kWh) | No |
| `battery_charge_kwh_std` | float | Std. deviation: battery throughput charged | No |
| `battery_discharge_kwh_mean` | float | Mean: battery throughput discharged (kWh) | No |
| `battery_discharge_kwh_std` | float | Std. deviation: battery throughput discharged | No |
| `battery_throughput_kwh_mean` | float | Mean: total energy movement (charge + discharge in kWh) | Yes – Devices |
| `battery_throughput_kwh_std` | float | Std. deviation: total energy movement | No |
| `battery_full_cycles_mean` | float | Mean: full equivalent cycles (IEEE standard) | Yes – Devices |
| `battery_full_cycles_std` | float | Std. deviation: full equivalent cycles | No |

---

## 2. `timelines.json`

**Created by:** `src/GridKIT/scripts/run_experiment.py` (batch sweep) and `src/GridKIT/scripts/train_run.py` (map UI runs, via `core.run_store.save_results()`)
**Used by:** Dashboard (Overload Map tab, line charts)

### JSON Structure

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

### Field Descriptions

| Field | Type | Description | Currently consumed by Dashboard |
|------|-----|--------------|-------------------------------|
| `penetration` | float | EV penetration level (0.0–1.0) | Yes – all tabs (scenario × penetration selection) |
| `scenario` | string | Scenario name | Yes – all tabs |
| `transformer_loading` | list[float] | Transformer loading per step (96 steps, p.u.) | Yes – Network utilization, Scenario comparison, Overview¹ |
| `max_line_loading` | list[float] | Maximum line loading per step (96 steps, p.u.) | Yes – Network utilization, Scenario comparison, Overview¹ |
| `curtailment` | list[bool] | Curtailment applied per step (96 steps) | Yes – Network utilization, Scenario comparison (shading) |
| `overloaded_transformers` | list[list[str]] | IDs of overloaded transformers per step (96 steps) | Yes – Network utilization (overload matrix) |
| `overloaded_lines` | list[list[str]] | IDs of overloaded lines per step (96 steps) | Yes – Network utilization (overload matrix) |
| `price` | list[float] | Electricity price per step (96 steps, €/kWh) | Yes – Scenario comparison (episode panel) |
| `base_load` | list[float] | Non-controllable load per step (96 steps, kW) | No |
| `pv_generation` | list[float] | PV generation per step (96 steps, kW) | Yes – Scenario comparison, Devices |
| `temperature` | list[float] | Outdoor temperature per step (96 steps, °C) | No |
| `ev_power` | list[float] | All EVs combined per step (96 steps, kW) | Yes – Devices, Scenario comparison |
| `battery_power` | list[float] | All batteries combined per step (96 steps, kW) | Yes – Devices, Scenario comparison |
| `hp_power` | list[float] | All heat pumps combined per step (96 steps, kW) | Yes – Devices, Scenario comparison |
| `house_ev_power` | list[float] | Single representative EV per step (96 steps, kW) | Yes – Devices (household view) |
| `house_ev_soc` | list[float] | Single representative EV SoC per step (96 steps, 0.0–1.0) | Yes – Devices (household view) |
| `house_ev_available` | list[float] | Single EV availability per step (96 steps, 0.0–1.0) | Yes – Devices (household view) |
| `house_battery_power` | list[float] | Single representative battery per step (96 steps, kW) | Yes – Devices (household view) |
| `house_battery_soc` | list[float] | Single representative battery SoC per step (96 steps, 0.0–1.0) | Yes – Devices (household view) |
| `house_battery_charge_cumulative_kwh` | list[float] | Single battery: cumulative charged energy (96 steps, kWh) – for full equivalent cycles | Yes – Devices (cumulative energy chart) |
| `house_battery_discharge_cumulative_kwh` | list[float] | Single battery: cumulative discharged energy (96 steps, kWh) – for full equivalent cycles | Yes – Devices (cumulative energy chart) |
| `house_hp_power` | list[float] | Single representative heat pump per step (96 steps, kW) | Yes – Devices (household view) |
| `house_hp_soc` | list[float] | Single representative HP SoC per step (96 steps, 0.0–1.0) | Yes – Devices (household view, comfort breaches) |
| `house_pv` | list[float] | Single representative PV system per step (96 steps, kW) | No |
| `soc_satisfaction_rate` | float | Fraction of agents at target SoC (0.0–1.0) | No |

¹ Overview reads `transformer_loading`/`max_line_loading` only as a fallback for older runs without `line_peak_max` or the overload steps breakdown in summary.json.

**Note:** 96 steps = 24 h × 4 steps/h (15-minute intervals)

---

## 3. `iteration_metrics.json`

**Created by:** `src/GridKIT/rl_engine/trainer.py` (invoked by `src/GridKIT/scripts/run_experiment.py` for the batch sweep and `src/GridKIT/scripts/train_run.py` for map UI runs)
**Used by:** Dashboard (Training tab)

**Note:** The file contains ~200 additional technical fields (config, timers, perf metrics). Only the fields relevant to the dashboard are documented.

### JSON Structure

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

### Field Descriptions

#### Basic Info

| Field | Type | Description | Currently consumed by Dashboard |
|------|-----|--------------|-------------------------------|
| `training_iteration` | int | Iteration number (1, 2, 3, ...) | Yes – Training (x-axis of return/entropy charts) |

#### Episode Rewards (`env_runners.*`)

| Field | Type | Description | Currently consumed by Dashboard |
|------|-----|--------------|-------------------------------|
| `episode_return_mean` | float | Average reward across all episodes in this iteration | Yes – Training (return chart) |
| `episode_return_min` | float | Minimum reward in this iteration | Yes – Training (return band) |
| `episode_return_max` | float | Maximum reward in this iteration | Yes – Training (return band) |
| `episode_len_mean` | float | Average episode length (steps) | No |

#### Policy Metrics (`learners.<policy>.*`)

| Field | Type | Description | Currently consumed by Dashboard |
|------|-----|--------------|-------------------------------|
| `policy_loss` | float | PPO clip loss of the policy | No |
| `entropy` | float | Policy entropy (exploration level) | Yes – Training (exploration per device type) |
| `vf_loss` | float | Value function loss | No |
| `vf_explained_var` | float | Variance of returns explained by the value function (higher = better) | No |

**Note:** `learners` contains the same structure for each policy: `ev_policy`, `battery_policy`, `hp_policy`

---

## File Paths in the Output Directory – gaining access to examplary files

```
outputs/{experiment_name}/
├── summary.json              # Aggregated metrics
├── timelines.json            # Time-series data
├── graphs/                   # PNGs from plot_results
└── checkpoints/
    ├── pen_20/
    │   └── iteration_metrics.json
    ├── pen_40/
    │   └── ...
    └── pen_60/
        └── ...
```

This is the batch-sweep layout written by `scripts/run_experiment.py`
(`--out`, default `outputs/`); `graphs/` is produced separately by
`scripts/plot_results.py`.

To gain examplary JSON-files, these runs can be triggerd headlessly, simply by running:

```
python src/GridKIT/scripts/run_experiment.py
```

It defaults to using the `/data/feeder_20.json` GridNetwork.

Runs started from the map UI store the same result files under `runs/<id>/` instead — see below.

---

## File Paths in the Run Directory (map UI pipeline)

The map UI does not write to `outputs/`. Every run started there lives in its
own directory under `runs/`, created by `core/run_store.py` and populated by
`scripts/train_run.py`:

```
runs/<run_id>/
├── config.json                  # run name, training/eval settings, timestamps
├── grid_network.json            # the GridNetwork built by the map UI
├── household_configuration.json # household device config (map_ui/household_config.py)
├── status.json                  # {state, progress, iteration, total_iters, message, updated}
├── summary.json                 # aggregated metrics (same schema as outputs/)
├── timelines.json               # representative 24 h time series (same schema as outputs/)
├── iteration_metrics.json       # RLlib training metrics (same schema, but at the run root)
└── checkpoints/                 # trained per-device RL policies (no metrics inside)
```

The three result files (`summary.json`, `timelines.json`,
`iteration_metrics.json`) use the exact same schemas as in the output directory.
To clarify: the layout is flatter because a UI-launched run is a single training run, not a
sweep.

- there is **no `pen_*` level** — one run trains the one device layout the map
  produced (its EV share is `configured_share(layout)`), so
  `iteration_metrics.json` sits directly next to `summary.json`/`timelines.json`,
  not under `checkpoints/pen_*/`;
- `checkpoints/` holds only the trained RL policies
  (`trainer.save_checkpoint()`), not the metrics;
- there are **no `graphs/`** — PNG plotting exists only for batch sweeps
  (`scripts/plot_results.py`).


---

*Last updated: 23 September 2026*
