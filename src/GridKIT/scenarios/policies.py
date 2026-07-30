# scenarios/policies.py
# ─────────────────────────────────────────────────────────────
# Non-learning customer behaviours for the scenario comparison.
# Imports ONLY from core — pure controllers driven by observations
# + the published day-ahead price.
#
# Multi-device: every household has three controllable device-agents
# (ev / battery / hp). A baseline sets the EV behaviour; the battery and heat
# pump use fixed rule-based controllers (the status quo of real home energy
# management), so the scenarios stay comparable to the learned policy.
#
#   Scenario 1 — NaiveImmediatePolicy   (EV: charge on arrival, flat tariff)
#   Scenario 2 — NaivePriceFollowPolicy (EV: cheapest window, jittered)
#   Battery — greedy self-consumption; Heat pump — thermostatic (both scenarios)
#
# Scenario 3 (selfish RL) is the trained IPPO policies, handled in rl_engine.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Protocol

import numpy as np

import core.constants as const
from core.models import BatteryAction, ChargingAction, HPAction, Observation, device_of


# Rule-based controller thresholds (baseline heuristics, not grid physics)
BATTERY_PV_CHARGE_KW = 0.1     # charge the battery when PV generation exceeds this
HP_THERMOSTAT_SETPOINT = 0.6   # heat while the thermal buffer is below this (> comfort floor)


class Policy(Protocol):
    """A customer behaviour: plan once per episode, then act each step.
    ``act`` returns a per-agent device-action *index* (decoded by the env)."""

    def reset(self, day_ahead_prices: list[float], rng: np.random.Generator) -> None: ...
    def act(self, observations: dict[str, Observation]) -> dict[str, int]: ...


# ── fixed device controllers (shared by all baselines) ───────
def _battery_greedy(obs: Observation) -> int:
    """Greedy self-consumption: soak up PV surplus, discharge to cover imports."""
    if obs.pv_generation_kw > BATTERY_PV_CHARGE_KW and obs.soc_progress < const.BATTERY_MAX_SOC:
        return int(BatteryAction.CHARGE)
    if obs.net_household_load_kw > 0.0 and obs.soc_progress > const.BATTERY_MIN_SOC:
        return int(BatteryAction.DISCHARGE)
    return int(BatteryAction.IDLE)


def _hp_thermostatic(obs: Observation) -> int:
    """Thermostat: run while the thermal buffer is below setpoint, ignore price/grid."""
    return int(HPAction.HEAT) if obs.soc_progress < HP_THERMOSTAT_SETPOINT else int(HPAction.OFF)


def _ev_charge_if_unfinished(obs: Observation, charge: bool) -> int:
    """EV: never charge past the SoC target (progress ≥ 1); OFF otherwise."""
    if charge and obs.soc_progress < 1.0:
        return int(ChargingAction.FULL)
    return int(ChargingAction.OFF)


def _dispatch(obs: Observation, ev_action: int) -> int:
    """Route one agent's observation to its device controller (EV set by the baseline)."""
    dev = device_of(obs.agent_id)
    if dev == const.DEVICE_EV:
        return ev_action
    if dev == const.DEVICE_BATTERY:
        return _battery_greedy(obs)
    return _hp_thermostatic(obs)   # DEVICE_HEAT_PUMP


class NaiveImmediatePolicy:
    """Scenario 1 — EV charges at full power whenever plugged in and below target
    (flat tariff, no shifting). Battery greedy-self-consumes; HP is thermostatic."""

    def reset(self, day_ahead_prices: list[float], rng: np.random.Generator) -> None:
        pass

    def act(self, observations: dict[str, Observation]) -> dict[str, int]:
        return {aid: _dispatch(o, _ev_charge_if_unfinished(o, charge=True))
                for aid, o in observations.items()}


class NaivePriceFollowPolicy:
    """Scenario 2 — EV waits for the cheapest contiguous window long enough to finish,
    then charges full until done. Battery/HP as in Scenario 1.

    `jitter_std` (steps) spreads the chosen start time: large σ ≈ manual/human,
    σ→0 ≈ automated/app-driven (every EV picks the same window → synchronization)."""

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
        remaining_frac = max(0.0, 1.0 - obs.soc_progress)
        energy_kwh = remaining_frac * const.EV_TARGET_SOC * const.EV_BATTERY_CAPACITY_KWH
        steps_needed = max(1, int(np.ceil(energy_kwh / (const.EV_POWER_FULL_KW * const.TIMESTEP_HOURS))))
        steps_needed = min(steps_needed, const.EPISODE_STEPS)
        kernel = np.ones(steps_needed)
        window_cost = np.convolve(self._prices, kernel, mode="valid")
        cheapest = int(np.argmin(window_cost))
        jittered = cheapest + int(round(self._rng.normal(0.0, self.jitter_std)))
        return int(np.clip(jittered, 0, const.EPISODE_STEPS - 1))

    def act(self, observations: dict[str, Observation]) -> dict[str, int]:
        actions: dict[str, int] = {}
        for aid, o in observations.items():
            if device_of(aid) == const.DEVICE_EV:
                if aid not in self._start_step:
                    self._start_step[aid] = self._plan_start(o)
                charge = self._t >= self._start_step[aid]
                actions[aid] = _dispatch(o, _ev_charge_if_unfinished(o, charge=charge))
            else:
                actions[aid] = _dispatch(o, ev_action=0)
        self._t += 1
        return actions
