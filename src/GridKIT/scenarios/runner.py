# scenarios/runner.py
# ─────────────────────────────────────────────────────────────
# Runs a customer behaviour against an injected environment and turns the
# per-step stream into EpisodeMetrics / SimResult. Aggregating over seeds
# gives the mean±std the analytical comparison needs (single-day numbers
# are anecdotes; the claim is about distributions).
#
# Imports ONLY from core (+ the policy Protocol) — the environment is
# injected as a GridEnvProtocol, so this never depends on grid_model.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import core.constants as const
from core.protocols import GridEnvProtocol
from core.models import EpisodeMetrics, PowerFlowResult, SimResult, bus_of, device_of
from scenarios.policies import Policy


def compute_episode_metrics(
    episode: int,
    timestep_results: list[PowerFlowResult],
    per_agent_return: dict[str, float],
    soc_satisfied: dict[str, bool],
    bill_by_household: dict[str, float] | None = None,
    hp_comfort_satisfaction_rate: float = 0.0,
    battery_charge_kwh: float = 0.0,
    battery_discharge_kwh: float = 0.0,
) -> EpisodeMetrics:
    """Reduce one episode's per-step stream + terminal flags into EpisodeMetrics."""
    curtailment_events = sum(1 for pf in timestep_results if pf.curtailment_applied)
    peak_loading = max((pf.transformer_loading_pu for pf in timestep_results), default=0.0)
    satisfaction = float(np.mean(list(soc_satisfied.values()))) if soc_satisfied else 0.0
    mean_return = float(np.mean(list(per_agent_return.values()))) if per_agent_return else 0.0

    # localize the stress: which feeder / line, how often, how bad
    feeder_steps: dict[str, int] = {}
    feeder_peak: dict[str, float] = {}
    line_steps: dict[str, int] = {}
    line_peak: dict[str, float] = {}
    for pf in timestep_results:
        for fid, load in pf.transformer_loadings_pu.items():
            feeder_peak[fid] = max(feeder_peak.get(fid, 0.0), load)
            feeder_steps.setdefault(fid, 0)
            if load > const.TRANSFORMER_OVERLOAD_THRESHOLD:
                feeder_steps[fid] += 1
        for lid, load in pf.line_loadings_pu.items():
            # peak from the watch level up (feeds the map's shading); step COUNT only for
            # genuine violations, so `line_overload_steps` stays a count of real trips
            if load >= const.LINE_WATCH_THRESHOLD:
                line_peak[lid] = max(line_peak.get(lid, 0.0), load)
            if load > const.LINE_OVERLOAD_THRESHOLD:
                line_steps[lid] = line_steps.get(lid, 0) + 1

    bills = list((bill_by_household or {}).values())
    return EpisodeMetrics(
        episode=episode,
        mean_episode_reward=mean_return,
        soc_satisfaction_rate=satisfaction,
        curtailment_events=curtailment_events,
        transformer_peak_loading_pu=peak_loading,
        mean_household_bill_eur=float(np.mean(bills)) if bills else 0.0,
        hp_comfort_satisfaction_rate=hp_comfort_satisfaction_rate,
        battery_charge_kwh=battery_charge_kwh,
        battery_discharge_kwh=battery_discharge_kwh,
        feeder_overload_steps=feeder_steps,
        feeder_peak_loading_pu=feeder_peak,
        line_overload_steps=line_steps,
        line_peak_loading_pu=line_peak,
    )


def run_episode(
    env: GridEnvProtocol,
    policy: Policy,
    seed: int,
    ev_penetration: float = 1.0,
) -> SimResult:
    """Play one full episode of `policy` on `env`, returning a populated SimResult."""
    obs = env.reset(seed=seed)
    policy.reset(env.day_ahead_prices(), np.random.default_rng(seed))

    timestep_results: list[PowerFlowResult] = []
    per_agent_return: dict[str, float] = {aid: 0.0 for aid in obs}
    final_soc: dict[str, float] = {}
    soc_satisfied: dict[str, bool] = {}
    bill_by_household: dict[str, float] = {}
    hp_comfort_steps = 0
    hp_total_steps = 0
    battery_charge_kwh = 0.0
    battery_discharge_kwh = 0.0

    while True:
        actions = policy.act(obs)
        step_results, power_flow = env.step(actions)
        timestep_results.append(power_flow)

        # the bill is a HOUSEHOLD quantity reported identically on each of that home's
        # device-agents, so count it once per house per step or it scales with device count
        counted: set[str] = set()
        for aid, result in step_results.items():
            house = bus_of(aid)
            if house not in counted and "bill_eur" in result.info:
                bill_by_household[house] = bill_by_household.get(house, 0.0) + result.info["bill_eur"]
                counted.add(house)

        # HP comfort is also a per-household quantity shared across that household's
        # device-agents (see bill above) — count it once per house per step.
        counted_hp: set[str] = set()
        for aid, result in step_results.items():
            house = bus_of(aid)
            if house not in counted_hp and "hp_thermal_soc" in result.info:
                hp_total_steps += 1
                if result.info["hp_thermal_soc"] >= const.HP_COMFORT_MIN_SOC:
                    hp_comfort_steps += 1
                counted_hp.add(house)

        # battery power is already a feeder-aggregate on the PowerFlowResult itself
        # (signed: + charge / − discharge) — read once per step, not per agent.
        batt_kw = power_flow.device_power_kw.get(const.DEVICE_BATTERY, 0.0)
        if batt_kw > 0:
            battery_charge_kwh += batt_kw * const.TIMESTEP_HOURS
        else:
            battery_discharge_kwh += -batt_kw * const.TIMESTEP_HOURS

        for aid, result in step_results.items():
            per_agent_return[aid] += result.reward
            # SoC verdict is meaningful only for EV agents (info is shared across a
            # household's device-agents, so guard by device type)
            if device_of(aid) == const.DEVICE_EV and result.info.get("ev_terminal"):
                final_soc[aid] = result.info["ev_final_soc"]
                soc_satisfied[aid] = result.info["ev_satisfied"]

        obs = {aid: r.observation for aid, r in step_results.items()}
        if any(r.done for r in step_results.values()):
            break

    hp_comfort_rate = (hp_comfort_steps / hp_total_steps) if hp_total_steps else 0.0
    metrics = compute_episode_metrics(seed, timestep_results, per_agent_return, soc_satisfied,
                                      bill_by_household, hp_comfort_rate,
                                      battery_charge_kwh, battery_discharge_kwh)
    return SimResult(
        episode=seed,
        network_id=env.network.network_id,
        ev_penetration=ev_penetration,
        timestep_results=timestep_results,
        final_soc_per_agent=final_soc,
        metrics=metrics,
    )


@dataclass
class ScenarioStats:
    """Aggregated results of one scenario over many seeds (the error bars)."""
    label: str
    n_episodes: int
    curtailment_events: tuple[float, float]       # (mean, std) timesteps with §14a dimming
    soc_satisfaction_rate: tuple[float, float]    # (mean, std) fraction meeting target
    transformer_peak_loading_pu: tuple[float, float]
    mean_episode_reward: tuple[float, float]
    mean_household_bill_eur: tuple[float, float] = (0.0, 0.0)   # € per household per 24 h
    hp_comfort_satisfaction_rate: tuple[float, float] = (0.0, 0.0)
    battery_charge_kwh: tuple[float, float] = (0.0, 0.0)
    battery_discharge_kwh: tuple[float, float] = (0.0, 0.0)
    # per-component stress, averaged over seeds — answers "which transformer / line was it"
    feeder_overload_steps: dict[str, tuple[float, float]] = field(default_factory=dict)
    feeder_peak_loading_pu: dict[str, tuple[float, float]] = field(default_factory=dict)
    line_overload_steps: dict[str, tuple[float, float]] = field(default_factory=dict)
    line_peak_loading_pu: dict[str, tuple[float, float]] = field(default_factory=dict)

    def worst_feeder(self) -> tuple[str, float, float] | None:
        """(feeder_id, mean overloaded steps, mean peak pu) for the most-stressed feeder.

        Ranked by how often it trips, then by how hard — the feeder to reinforce first.
        """
        if not self.feeder_overload_steps:
            return None
        fid = max(self.feeder_overload_steps,
                  key=lambda f: (self.feeder_overload_steps[f][0],
                                 self.feeder_peak_loading_pu.get(f, (0.0, 0.0))[0]))
        return fid, self.feeder_overload_steps[fid][0], self.feeder_peak_loading_pu.get(fid, (0.0, 0.0))[0]

    def __str__(self) -> str:
        def ms(x):
            return f"{x[0]:.2f}±{x[1]:.2f}"
        worst = self.worst_feeder()
        culprit = ""
        if worst and worst[1] > 0:
            culprit = f"  worst={worst[0]}({worst[1]:.1f} steps @ {worst[2]:.2f}pu)"
        return (
            f"{self.label:<28} curtail={ms(self.curtailment_events):>12}  "
            f"soc_ok={ms(self.soc_satisfaction_rate):>10}  "
            f"peak_pu={ms(self.transformer_peak_loading_pu):>10}  "
            f"reward={ms(self.mean_episode_reward):>12}  "
            f"bill={ms(self.mean_household_bill_eur):>12}  "
            f"hp_ok={ms(self.hp_comfort_satisfaction_rate):>10}  "
            f"batt_kwh(+/-)={ms(self.battery_charge_kwh)}/{ms(self.battery_discharge_kwh)}{culprit}"
        )


def run_scenario(
    env: GridEnvProtocol,
    policy: Policy,
    seeds: list[int],
    label: str = "scenario",
    ev_penetration: float = 1.0,
) -> ScenarioStats:
    """Run `policy` over `seeds` and aggregate the per-episode metrics into mean±std."""
    curt, soc, peak, rew, bill = [], [], [], [], []
    hp_comfort, batt_charge, batt_discharge = [], [], []
    per_episode: list[EpisodeMetrics] = []
    for seed in seeds:
        result = run_episode(env, policy, seed, ev_penetration)
        m = result.metrics
        per_episode.append(m)
        curt.append(m.curtailment_events)
        soc.append(m.soc_satisfaction_rate)
        peak.append(m.transformer_peak_loading_pu)
        rew.append(m.mean_episode_reward)
        bill.append(m.mean_household_bill_eur)
        hp_comfort.append(m.hp_comfort_satisfaction_rate)
        batt_charge.append(m.battery_charge_kwh)
        batt_discharge.append(m.battery_discharge_kwh)

    def ms(xs):
        return (float(np.mean(xs)), float(np.std(xs)))

    def by_component(attr: str) -> dict[str, tuple[float, float]]:
        """mean±std per component, counting a seed that never listed it as a zero.

        A line that trips in one seed out of six really did average 1/6 of its trips —
        dropping the silent seeds would inflate every per-component number.
        """
        keys = {k for m in per_episode for k in getattr(m, attr)}
        return {k: ms([getattr(m, attr).get(k, 0) for m in per_episode]) for k in sorted(keys)}

    return ScenarioStats(
        label=label,
        n_episodes=len(seeds),
        curtailment_events=ms(curt),
        soc_satisfaction_rate=ms(soc),
        transformer_peak_loading_pu=ms(peak),
        mean_episode_reward=ms(rew),
        mean_household_bill_eur=ms(bill),
        hp_comfort_satisfaction_rate=ms(hp_comfort),
        battery_charge_kwh=ms(batt_charge),
        battery_discharge_kwh=ms(batt_discharge),
        feeder_overload_steps=by_component("feeder_overload_steps"),
        feeder_peak_loading_pu=by_component("feeder_peak_loading_pu"),
        line_overload_steps=by_component("line_overload_steps"),
        line_peak_loading_pu=by_component("line_peak_loading_pu"),
    )
