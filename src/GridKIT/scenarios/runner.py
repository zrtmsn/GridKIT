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

from dataclasses import dataclass

import numpy as np

from core.protocols import GridEnvProtocol
from core.models import EpisodeMetrics, PowerFlowResult, SimResult
from scenarios.policies import Policy


def compute_episode_metrics(
    episode: int,
    timestep_results: list[PowerFlowResult],
    per_agent_return: dict[str, float],
    soc_satisfied: dict[str, bool],
) -> EpisodeMetrics:
    """Reduce one episode's per-step stream + terminal flags into EpisodeMetrics."""
    curtailment_events = sum(1 for pf in timestep_results if pf.curtailment_applied)
    peak_loading = max((pf.transformer_loading_pu for pf in timestep_results), default=0.0)
    satisfaction = float(np.mean(list(soc_satisfied.values()))) if soc_satisfied else 0.0
    mean_return = float(np.mean(list(per_agent_return.values()))) if per_agent_return else 0.0
    return EpisodeMetrics(
        episode=episode,
        mean_episode_reward=mean_return,
        soc_satisfaction_rate=satisfaction,
        curtailment_events=curtailment_events,
        transformer_peak_loading_pu=peak_loading,
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

    while True:
        actions = policy.act(obs)
        step_results, power_flow = env.step(actions)
        timestep_results.append(power_flow)

        for aid, result in step_results.items():
            per_agent_return[aid] += result.reward
            if result.info.get("terminal"):
                final_soc[aid] = result.info["final_soc"]
                soc_satisfied[aid] = result.info["soc_satisfied"]

        obs = {aid: r.observation for aid, r in step_results.items()}
        if any(r.done for r in step_results.values()):
            break

    metrics = compute_episode_metrics(seed, timestep_results, per_agent_return, soc_satisfied)
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

    def __str__(self) -> str:
        def ms(x):
            return f"{x[0]:.2f}±{x[1]:.2f}"
        return (
            f"{self.label:<28} curtail={ms(self.curtailment_events):>12}  "
            f"soc_ok={ms(self.soc_satisfaction_rate):>10}  "
            f"peak_pu={ms(self.transformer_peak_loading_pu):>10}  "
            f"reward={ms(self.mean_episode_reward):>12}"
        )


def run_scenario(
    env: GridEnvProtocol,
    policy: Policy,
    seeds: list[int],
    label: str = "scenario",
    ev_penetration: float = 1.0,
) -> ScenarioStats:
    """Run `policy` over `seeds` and aggregate the per-episode metrics into mean±std."""
    curt, soc, peak, rew = [], [], [], []
    for seed in seeds:
        result = run_episode(env, policy, seed, ev_penetration)
        m = result.metrics
        curt.append(m.curtailment_events)
        soc.append(m.soc_satisfaction_rate)
        peak.append(m.transformer_peak_loading_pu)
        rew.append(m.mean_episode_reward)

    def ms(xs):
        return (float(np.mean(xs)), float(np.std(xs)))

    return ScenarioStats(
        label=label,
        n_episodes=len(seeds),
        curtailment_events=ms(curt),
        soc_satisfaction_rate=ms(soc),
        transformer_peak_loading_pu=ms(peak),
        mean_episode_reward=ms(rew),
    )
