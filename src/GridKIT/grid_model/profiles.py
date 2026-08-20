# grid_model/profiles.py
# ─────────────────────────────────────────────────────────────
# Time-series inputs for one episode: the published day-ahead price
# curve and the inflexible (BDEW-H0-like) household base load.
#
# Both are deterministic given the rng, so an episode is fully
# reproducible from its seed. Shapes are academic stand-ins, not
# calibrated data — see project scope notes.
# ─────────────────────────────────────────────────────────────
import numpy as np

import core.constants as const


def hour_to_step(hour: float) -> int:
    """
    Map a wall-clock hour (0–24) to an episode-local step index.

    The episode starts at EPISODE_START_HOUR (noon), so the overnight
    connected window (evening arrival → morning departure) does not wrap.
    e.g. 18:00 → step 24, 07:00 → step 76.
    """
    steps_per_hour = 60 / const.TIMESTEP_MINUTES
    return int(round(((hour - const.EPISODE_START_HOUR) % 24) * steps_per_hour))


def _episode_hours() -> np.ndarray:
    """Wall-clock hour at each of the EPISODE_STEPS timesteps."""
    steps = np.arange(const.EPISODE_STEPS)
    return (const.EPISODE_START_HOUR + steps * const.TIMESTEP_HOURS) % 24


def _bell(hours: np.ndarray, centre: float, width: float) -> np.ndarray:
    """Gaussian bump centred on a wall-clock hour, wrap-aware (24 h)."""
    d = (hours - centre + 12) % 24 - 12   # signed hour distance in [-12, 12)
    return np.exp(-0.5 * (d / width) ** 2)


def price_profile(rng: np.random.Generator, scenario: str = "medium") -> np.ndarray:
    """
    Published day-ahead price (€/kWh) per step: cheap overnight, expensive
    morning/evening. The overnight trough is the window naive price-followers
    synchronize into — and it falls inside the EV connected window.
    """
    hours = _episode_hours()
    # +morning peak, +evening peak, −overnight trough → recognizable day-ahead shape
    shape = (
        _bell(hours, centre=8.0, width=2.0) * 0.7
        + _bell(hours, centre=19.0, width=3.0) * 1.0
        - _bell(hours, centre=3.0, width=3.0) * 1.0
    )
    mult = const.PRICE_SCENARIO_MULTIPLIER.get(scenario, 1.0)
    price = const.PRICE_BASE_EUR_KWH * mult + const.PRICE_PEAK_AMPLITUDE_EUR_KWH * mult * shape
    price = price + rng.normal(0.0, const.PRICE_NOISE_STD_EUR_KWH, size=price.shape)
    return np.clip(price, 0.01, None)


def base_load_profile(rng: np.random.Generator) -> np.ndarray:
    """
    Inflexible household base load (kW) per step, BDEW-H0-like: low overnight,
    a morning bump and an evening peak. Scaled by a per-episode multiplier so
    grid stress varies across episodes.
    """
    hours = _episode_hours()
    shape01 = (
        _bell(hours, centre=7.0, width=2.0) * 0.6
        + _bell(hours, centre=19.0, width=3.0) * 1.0
    )
    shape01 = shape01 / shape01.max()   # normalize peak to 1.0
    span = const.HOUSEHOLD_BASE_LOAD_PEAK_KW - const.HOUSEHOLD_BASE_LOAD_MEAN_KW
    load = const.HOUSEHOLD_BASE_LOAD_MEAN_KW + span * shape01
    multiplier = rng.uniform(const.LOAD_MULTIPLIER_MIN, const.LOAD_MULTIPLIER_MAX)
    return load * multiplier
