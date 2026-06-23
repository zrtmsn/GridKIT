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

## Results

Full run: 3 EV penetrations × 4 behaviours × 12 seeds; scenario 3 trained 60 IPPO
iterations per penetration (reward converged ~3 → ~105). **Curtailment events** = timesteps
per 24 h episode where §14a dimming triggered.

| Penetration | Behaviour | Curtailment events | SoC met |
|---|---|---:|---:|
| **20%** | all four | ~0 | grid copes |
| **40%** | flat / immediate | 19.9 ± 2.8 | 1.00 |
|         | price-follow manual (σ=8) | **3.8 ± 4.3** | 0.80 |
|         | price-follow automated (σ=0) | **18.2 ± 4.4** | 0.97 |
|         | selfish RL | **9.4 ± 5.1** | 1.00 |
| **60%** | flat / immediate | 29.9 ± 2.5 | 1.00 |
|         | price-follow manual | 19.8 ± 5.7 | 0.58 |
|         | price-follow automated | 25.8 ± 1.3 | **0.27** |
|         | selfish RL | 27.5 ± 4.2 | 0.98 |

### Interpretation

1. **The synchronization risk is real and automation-driven.** At 40%, moving from manual
   to automated price-following spikes curtailment **3.8 → 18.2** — same behaviour, just
   less jitter. Synchronization, not price-reactivity per se, is the failure mode.

2. **Smarter selfish behaviour helps — when avoiding congestion also serves the agent.**
   At 40% the RL agent cuts curtailment vs naive automated (**18.2 → 9.4**) *while* keeping
   everyone charged (SoC 1.00 vs 0.97). Grid-protective behaviour emerged as a side effect
   of pure self-interest.

3. **Under heavy load, self-interest stops protecting the grid.** At 60% the RL agent's
   curtailment climbs to 27.5 — nearly as high as flat/immediate — because the §14a 4.2 kW
   floor guarantees it enough power that charging *through* congestion (SoC 0.98) beats
   missing its deadline. Meanwhile naive automated collapses to 27% SoC satisfaction.

**Takeaway:** grid-friendliness emerges from self-interest only at moderate stress; under
heavy load it does not. §14a remains necessary as a backstop, and the curtailment-vs-SoC
trade-off is exactly what a regulator weighs.

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
