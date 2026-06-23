# scenarios/policies.py
# ─────────────────────────────────────────────────────────────
# Non-learning customer behaviours for the scenario comparison.
# Imports ONLY from core — these are pure controllers driven by
# observations + the published day-ahead price.
#
#   Scenario 1 — NaiveImmediatePolicy   (flat tariff, charge on arrival)
#   Scenario 2 — NaivePriceFollowPolicy (charge in the cheapest window, jittered)
#
# Scenario 3 (selfish RL) is the trained IPPO policy, handled in rl_engine.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Protocol

import numpy as np

import core.constants as const
from core.models import ChargingAction, Observation


class Policy(Protocol):
    """A customer behaviour: plan once per episode, then act each step."""

    def reset(self, day_ahead_prices: list[float], rng: np.random.Generator) -> None: ...
    def act(self, observations: dict[str, Observation]) -> dict[str, ChargingAction]: ...


def _charge_if_unfinished(obs: Observation, charge: bool) -> ChargingAction:
    """Common rule: never charge past the SoC target; OFF when not charging."""
    if charge and obs.soc_progress < 1.0:
        return ChargingAction.FULL
    return ChargingAction.OFF


class NaiveImmediatePolicy:
    """
    Scenario 1 — flat tariff, no incentive to shift: charge at full power whenever
    plugged in and below target. (The environment forces OFF while disconnected.)
    """

    def reset(self, day_ahead_prices: list[float], rng: np.random.Generator) -> None:
        pass

    def act(self, observations: dict[str, Observation]) -> dict[str, ChargingAction]:
        return {aid: _charge_if_unfinished(o, charge=True) for aid, o in observations.items()}


class NaivePriceFollowPolicy:
    """
    Scenario 2 — volatile pricing, dumb-but-reactive: wait for the cheapest contiguous
    price window long enough to finish, then charge at full power until done.

    Every household follows the *same* published curve, so without noise they all pick
    the same window — the synchronization risk. `jitter_std` (steps) spreads the chosen
    start time: large σ ≈ manual/human behaviour, σ→0 ≈ automated/app-driven.
    """

    def __init__(self, jitter_std: float = const.NAIVE_PRICE_JITTER_STEPS_STD):
        self.jitter_std = jitter_std
        self._prices = np.zeros(const.EPISODE_STEPS)
        self._rng = np.random.default_rng()
        self._t = 0
        self._start_step: dict[str, int] = {}

    def reset(self, day_ahead_prices: list[float], rng: np.random.Generator) -> None:
        self._prices = np.asarray(day_ahead_prices, dtype=float)
        self._rng = rng
        self._t = 0
        self._start_step = {}

    def _plan_start(self, obs: Observation) -> int:
        """Cheapest contiguous window for this agent's remaining energy need, + jitter."""
        remaining_frac = max(0.0, 1.0 - obs.soc_progress)          # fraction of target still to go
        energy_kwh = remaining_frac * const.EV_TARGET_SOC * const.EV_BATTERY_CAPACITY_KWH
        steps_needed = max(1, int(np.ceil(energy_kwh / (const.EV_POWER_FULL_KW * const.TIMESTEP_HOURS))))
        steps_needed = min(steps_needed, const.EPISODE_STEPS)

        # moving-sum of price over a window of steps_needed → cheapest start
        kernel = np.ones(steps_needed)
        window_cost = np.convolve(self._prices, kernel, mode="valid")
        cheapest = int(np.argmin(window_cost))

        jittered = cheapest + int(round(self._rng.normal(0.0, self.jitter_std)))
        return int(np.clip(jittered, 0, const.EPISODE_STEPS - 1))

    def act(self, observations: dict[str, Observation]) -> dict[str, ChargingAction]:
        actions: dict[str, ChargingAction] = {}
        for aid, o in observations.items():
            if aid not in self._start_step:
                self._start_step[aid] = self._plan_start(o)
            charge = self._t >= self._start_step[aid]
            actions[aid] = _charge_if_unfinished(o, charge=charge)
        self._t += 1
        return actions
