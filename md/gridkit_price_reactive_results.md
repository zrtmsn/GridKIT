# GridKIT — Price-Reactive Charging & §14a: Changes and Results

Branch: `feature/price-reactive-scenarios`

## The question

§14a EnWG lets grid operators curtail controllable EV charging when a low-voltage
feeder is at risk of overload. It was designed for flat tariffs and uncoordinated
charging. **Does it still hold up once customers become price-reactive — and does
smarter (RL) charging make that better or worse?**

We compare four customer behaviours on the same feeder, across EV penetration levels:

1. **Flat / immediate** — flat tariff, charge on arrival (today's baseline).
2. **Price-follow, manual** — charge in the cheapest window, with wide start-time jitter (σ=8).
3. **Price-follow, automated** — same, but app-driven (σ→0), everyone hits the same window.
4. **Selfish RL** — a learned, purely self-interested policy that can *sense* local
   congestion (lagged voltage / past dimming) but is never rewarded for protecting the grid.

---

## Commit 1 — environment realism & scenarios (`168dda4`)

Made the simulation able to exhibit the phenomenon at all. Before this, every agent
was identical (fixed arrival/departure, mean SoC, constant price and 11 kW base load),
forcing lockstep behaviour and degenerating the comparison.

- **Heterogeneous, seeded EVs** (`grid_model/ev_sampler.py`) — per-episode arrival /
  departure / SoC, so agents desynchronize via private state.
- **Dynamic inputs** (`grid_model/profiles.py`) — day-ahead-like price (cheap overnight)
  and BDEW-H0-like base load; `GridEnv.day_ahead_prices()` exposes the forecast.
- **Noon-shifted episode** (`EPISODE_START_HOUR=12`) so the overnight connected window
  no longer wraps — this unblocked realistic arrival/departure sampling.
- **Selfish congestion-aware observation** (`OBS_DIM` 5→7): added lagged, smart-meter-
  observable `local_voltage_pu` and `recent_curtailment_ratio`, giving a self-interested
  agent a channel to perceive and avoid grid stress — without any cooperative reward term.
- **§14a as proportional dimming with a 4.2 kW floor** (was a flat cap); nonlinear PyPSA
  pf for physical voltages.
- **Scenario harness** (`scenarios/`) — non-RL policies + a runner producing
  `EpisodeMetrics` (curtailment events, SoC satisfaction, peak loading) as mean±std over seeds.
- Fixes: duplicate EV loads removed; transformer impedance added; stub re-sized so it can
  actually be stressed.

## Commit 2 — full end-to-end pipeline & app

Turned the prototype into a runnable experiment + dashboard.

- **Power-flow surrogate** (`grid_model/surrogate.py`) — `RadialPowerFlow`: exact line/
  transformer flows + linearized voltage drop for the radial feeder. **Validated within
  ~1.5% of PyPSA** (`test_surrogate.py`) and **~3700× faster** (episode 11.3 s → 0.003 s).
  Default backend; PyPSA pf kept as a validation option (`use_surrogate=False`).
- **EV penetration** — `GridEnv(ev_penetration=…)`; EVs spread evenly across the feeder.
- **RL adapter + checkpointing** (`rl_engine/rl_policy.py`, `obs_norm.py`, `trainer.py`) —
  a trained IPPO policy is wrapped as a `scenarios.Policy` and run through the *same*
  runner as the naive behaviours, so scenario 3 yields comparable metrics. It **samples**
  per agent (not argmax), so identical agents still desynchronize.
- **Experiment runner** (`scripts/run_experiment.py`) — trains scenario 3 per penetration,
  evaluates scenarios 1–3 over seeds, saves `outputs/summary.json` + `outputs/timelines.json`.
- **Streamlit dashboard** (`dashboard/app.py`) — curtailment comparison across penetration,
  SoC-satisfaction table, representative 24 h loading/curtailment + price/base timelines.
- **Orchestration** — `scripts/app.py` (`gridkit-app --train`) and `scripts/fetch_network.py`
  (OSM → JSON via map_ui; `GridEnv` now takes an injectable builder).
- **20-household feeder** (`data/feeder_20.json`, trafo 0.07 MVA) sized so the penetration
  axis is meaningful. Small `stub_network.json` kept for fast tests.
- Declared `torch` and `streamlit` in `pyproject.toml`.

---

## Calibration note (energy need)

The first run used `EV_INITIAL_SOC_MEAN = 0.30` (every car ~empty daily ≈ 200 km/day),
which forced *every* EV into a ~5 h full charge and made the flat/immediate baseline overload
all evening (≈30 curtailment events at 60%) — implausibly high for an uncoordinated baseline.
We recalibrated to realistic daily depletion (`EV_INITIAL_SOC_MEAN = 0.55` ≈ 19 kWh/night
≈ 110 km/day) and a wider home-arrival spread (`EV_ARRIVAL_HOUR_STD` 1.0 → 1.5 h). The energy
need — not arrival clustering — was the dominant driver. Numbers below are the recalibrated run.

## Results

Full run: 3 EV penetrations × 4 behaviours × 12 seeds; scenario 3 trained 60 IPPO
iterations per penetration. **Curtailment events** = timesteps per 24 h episode where §14a
dimming triggered.

| Penetration | Behaviour | Curtailment events | SoC met |
|---|---|---:|---:|
| **20%** | all four | ~0 | grid copes |
| **40%** | flat / immediate | 6.8 ± 3.7 | 1.00 |
|         | price-follow manual (σ=8) | **0.0 ± 0.0** | 0.93 |
|         | price-follow automated (σ=0) | **5.8 ± 3.7** | 1.00 |
|         | selfish RL | **0.2 ± 0.4** | 0.97 |
| **60%** | flat / immediate | 13.5 ± 3.6 | 1.00 |
|         | price-follow manual | 2.7 ± 3.4 | 0.93 |
|         | price-follow automated | **13.6 ± 2.1** | 0.99 |
|         | selfish RL | **1.4 ± 1.3** | 1.00 |

### Interpretation

1. **The synchronization risk is real and automation-driven.** At 60%, moving from manual
   (σ=8) to automated (σ→0) price-following jumps curtailment **2.7 → 13.6** — same behaviour,
   just less jitter. Synchronization, not price-reactivity per se, is the failure mode.

2. **Where charging lands matters as much as how much.** Naive *immediate* charging stacks on
   the evening base-load peak (flat/immediate ≈ 6.8 / 13.5), whereas price-following shifts
   load into the overnight base-load trough — so even synchronized automated price-following
   is no worse than naive immediate, and *spread* price-following is far gentler.

3. **Congestion-aware self-interest nearly eliminates curtailment.** The selfish RL agent —
   never rewarded for protecting the grid, only able to *sense* local voltage / past dimming —
   drives curtailment to **0.2 (40%) / 1.4 (60%)** while still charging everyone (SoC ~1.0).
   With realistic energy needs there is enough slack to both dodge congestion and meet the
   deadline, and pure self-interest discovers it.

**Takeaway:** automated price-following is the genuine new risk §14a must withstand; a
self-interested agent that can perceive local congestion resolves it without any
grid-protective reward — *provided there is charging slack*. (An earlier tight-energy
calibration showed the opposite regime: when every car must charge ~5 h, self-interest charges
through congestion and §14a stays the binding backstop.) Both regimes argue the same thing —
the §14a curtailment mechanism is what makes either outcome safe.

### Scope / caveats

Demonstrated mechanism under named simplifications — a *relative* comparison, **not** a
real-grid forecast. Stub/synthetic feeder, surrogate (loss-free) power flow, synthetic
price/base-load profiles, shared-policy IPPO. Deferred: cooperative scenario 4 (social-
optimum upper bound), line-localized curtailment, real OSM topology runs (one command away
via `fetch_network.py`).

---

## Reproduce

```bash
# full experiment (writes outputs/) + dashboard
gridkit-app --train                 # or: python -m GridKIT.scripts.app --train
# experiment only
python -m GridKIT.scripts.run_experiment --iterations 60 --seeds 12
# non-RL σ-sweep (scenarios 1 vs 2)
python -m GridKIT.scripts.run_scenarios --seeds 8
# tests
pytest src/GridKIT/grid_model src/GridKIT/scenarios -q
# real Karlsruhe topology
python -m GridKIT.scripts.fetch_network --out data/karlsruhe.json
python -m GridKIT.scripts.run_experiment --network data/karlsruhe.json
```
