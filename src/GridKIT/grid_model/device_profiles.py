# grid_model/device_profiles.py
# ─────────────────────────────────────────────────────────────
# Real per-household device time series for one episode, sourced from
# GridCreator's pyCity generator functions instead of the synthetic
# samplers in profiles.py / ev_sampler.py.
#
# What this replaces with physics-based, weather-driven data:
#   - household base load      (pyCity stochastic appliance model)
#   - EV availability window    (occupancy-driven)
#   - heat-pump electric load   (ambient-temperature + COP driven)
#   - PV generation             (weather + tilt/orientation)
#   - outdoor temperature        (real TRY series; was hardcoded 20 °C)
#
# Why a cached pool:
#   pyCity simulates a full stochastic YEAR per household — far too slow to
#   run per episode during RL training. So we build a small annual pool ONCE
#   (cached to disk), then each episode cheaply slices a noon-to-noon 24 h day
#   and resamples the hourly data to the 96 fifteen-minute steps GridKIT uses.
#
# Data source: pyCity's *bundled* TRY weather — needs NO GridCreator input/
# download and NO ding0. See vendor/GridCreator/creating_demand_and_load.py.
#
# Scope: this module produces only EXOGENOUS time series. Stateful/controllable
# device state (EV SoC, battery, HP thermal store) lives in core models + env.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import core.constants as const

# Annual pyCity horizon (hourly, one year).
_ANNUAL_HOURS = 8760
_STEPS_PER_HOUR = int(round(60 / const.TIMESTEP_MINUTES))   # 4 at 15 min
# Latest noon-start day whose 24 h window stays inside the year.
_MAX_START_DAY = (_ANNUAL_HOURS - 24) // 24 - 1             # 363


# ══════════════════════════════════════════════════════════════
# Per-episode, per-household output
# ══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class HouseholdDayProfiles:
    """One household's exogenous device profiles for a single episode.

    All arrays are length EPISODE_STEPS (96), aligned to the noon-start,
    15-minute GridKIT episode clock. Powers are in kW.
    """
    base_load_kw: np.ndarray     # inflexible household appliance load
    ev_available: np.ndarray     # 1.0 when the EV is parked/plugged-in, else 0.0
    hp_load_kw: np.ndarray       # heat-pump electric draw (its exogenous demand)
    pv_gen_kw: np.ndarray        # PV generation for this household's array
    temperature_c: np.ndarray    # outdoor ambient temperature

    def __post_init__(self) -> None:
        for name in ("base_load_kw", "ev_available", "hp_load_kw", "pv_gen_kw", "temperature_c"):
            arr = getattr(self, name)
            if arr.shape != (const.EPISODE_STEPS,):
                raise ValueError(f"{name} must have shape ({const.EPISODE_STEPS},), got {arr.shape}")


# ══════════════════════════════════════════════════════════════
# Pure slicing / resampling helpers (no pyCity — unit-testable)
# ══════════════════════════════════════════════════════════════
def _hourly_to_steps(hourly_window: np.ndarray, *, hold: bool) -> np.ndarray:
    """Resample a 24-value hourly window onto the 96 episode steps.

    hold=True  → piecewise-constant (each hour repeated STEPS_PER_HOUR times);
                 use for binary/availability signals.
    hold=False → linear interpolation between hourly samples; use for continuous
                 quantities (loads, PV, temperature).
    """
    if hourly_window.shape != (24,):
        raise ValueError(f"expected a 24 h window, got {hourly_window.shape}")
    if hold:
        return np.repeat(hourly_window, _STEPS_PER_HOUR)[: const.EPISODE_STEPS].astype(float)
    xp = np.arange(24) * _STEPS_PER_HOUR          # hourly sample positions: 0,4,…,92
    x = np.arange(const.EPISODE_STEPS)            # 0…95 (last 3 flat-extrapolate)
    return np.interp(x, xp, hourly_window).astype(float)


def _noon_window(annual: np.ndarray, start_day: int) -> np.ndarray:
    """Extract the 24 hourly values of the noon→noon window for a given day."""
    start = start_day * 24 + const.EPISODE_START_HOUR
    return np.asarray(annual[start : start + 24], dtype=float)


# ══════════════════════════════════════════════════════════════
# Provider
# ══════════════════════════════════════════════════════════════
class DeviceProfileProvider:
    """Holds an annual pool of device profiles and samples episode-days from it.

    The pool is a set of hourly annual arrays:
      base_load_sims   (n_sims, 8760)   heterogeneous per-household appliance load
      ev_avail_sims    (n_sims, 8760)   occupancy-driven availability (0/1)
      hp_annual        (8760,)          shared, weather-driven heat-pump load
      pv_annual_per_kwp(8760,)          shared PV yield per installed kWp
      temp_annual      (8760,)          shared outdoor temperature

    HP/PV/temperature are shared across the feeder (same neighborhood weather);
    base load and EV availability vary per household (different occupancy draws).

    Construct directly from arrays (tests / injection), or via
    ``DeviceProfileProvider.build(...)`` which loads a disk cache or generates
    the pool from pyCity.
    """

    def __init__(
        self,
        *,
        base_load_sims: np.ndarray,
        ev_avail_sims: np.ndarray,
        hp_annual: np.ndarray,
        pv_annual_per_kwp: np.ndarray,
        temp_annual: np.ndarray,
        pv_kwp_range: tuple[float, float] = (3.0, 10.0),
    ):
        self.base_load_sims = np.asarray(base_load_sims, dtype=float)
        self.ev_avail_sims = np.asarray(ev_avail_sims, dtype=float)
        self.hp_annual = np.asarray(hp_annual, dtype=float)
        self.pv_annual_per_kwp = np.asarray(pv_annual_per_kwp, dtype=float)
        self.temp_annual = np.asarray(temp_annual, dtype=float)
        self.pv_kwp_range = pv_kwp_range

        n_sims, hours = self.base_load_sims.shape
        if self.ev_avail_sims.shape != (n_sims, hours):
            raise ValueError("ev_avail_sims must match base_load_sims shape")
        for name, arr in (("hp_annual", self.hp_annual),
                          ("pv_annual_per_kwp", self.pv_annual_per_kwp),
                          ("temp_annual", self.temp_annual)):
            if arr.shape != (hours,):
                raise ValueError(f"{name} must have shape ({hours},), got {arr.shape}")
        self._n_sims = n_sims

    @property
    def n_sims(self) -> int:
        return self._n_sims

    # ── sampling ─────────────────────────────────────────────
    def sample_episode(self, n_households: int, rng: np.random.Generator) -> list[HouseholdDayProfiles]:
        """Sample profiles for a full feeder for one episode.

        One weather day is drawn for the whole feeder (physically consistent),
        then each household draws its own occupancy sim and PV size.
        """
        start_day = int(rng.integers(0, _MAX_START_DAY + 1))
        hp = _hourly_to_steps(_noon_window(self.hp_annual, start_day), hold=False)
        pv_per_kwp = _hourly_to_steps(_noon_window(self.pv_annual_per_kwp, start_day), hold=False)
        temp = _hourly_to_steps(_noon_window(self.temp_annual, start_day), hold=False)

        out: list[HouseholdDayProfiles] = []
        for _ in range(n_households):
            sim = int(rng.integers(0, self._n_sims))
            kwp = float(rng.uniform(*self.pv_kwp_range))
            base = _hourly_to_steps(_noon_window(self.base_load_sims[sim], start_day), hold=False)
            avail = _hourly_to_steps(_noon_window(self.ev_avail_sims[sim], start_day), hold=True)
            out.append(
                HouseholdDayProfiles(
                    base_load_kw=np.clip(base, 0.0, None),
                    ev_available=(avail >= 0.5).astype(float),
                    hp_load_kw=np.clip(hp, 0.0, None),
                    pv_gen_kw=np.clip(pv_per_kwp * kwp, 0.0, None),
                    temperature_c=temp,
                )
            )
        return out

    # ── construction with disk cache ─────────────────────────
    @classmethod
    def build(
        cls,
        *,
        n_household_sims: int = 8,
        persons_choices: tuple[int, ...] = (1, 2, 3, 4, 5),
        pv_ref_kwp: float = 5.0,
        pv_tilt_deg: float = 35.0,
        pv_orientation: str = "Süd",       # NOTE: GridCreator's solar_dict uses GERMAN keys
        ev_home_requires_all: bool = False,  # False = car parked whenever anyone is home
        seed: int = 0,
        cache_dir: str | Path = "cache",
        rebuild: bool = False,
    ) -> "DeviceProfileProvider":
        """Load the annual pool from cache, or generate it via pyCity and cache it."""
        params = dict(
            n_household_sims=n_household_sims,
            persons_choices=list(persons_choices),
            pv_ref_kwp=pv_ref_kwp,
            pv_tilt_deg=pv_tilt_deg,
            pv_orientation=pv_orientation,
            ev_home_requires_all=ev_home_requires_all,
            seed=seed,
            v=1,
        )
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        cache_path = Path(cache_dir) / f"device_profiles_{key}.npz"

        if cache_path.exists() and not rebuild:
            data = np.load(cache_path)
            return cls(
                base_load_sims=data["base_load_sims"],
                ev_avail_sims=data["ev_avail_sims"],
                hp_annual=data["hp_annual"],
                pv_annual_per_kwp=data["pv_annual_per_kwp"],
                temp_annual=data["temp_annual"],
            )

        pool = _generate_pool_pycity(**params)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **pool)
        return cls(**pool)


# ══════════════════════════════════════════════════════════════
# pyCity pool generation (isolated so the rest stays import-light)
# ══════════════════════════════════════════════════════════════
def _make_pycity_environment():
    """Build a pyCity Environment from the BUNDLED TRY weather (no external data)."""
    from pycity_base.classes.timer import Timer
    from pycity_base.classes.weather import Weather
    from pycity_base.classes.prices import Prices
    from pycity_base.classes.environment import Environment

    timer = Timer(
        time_discretization=3600,
        timesteps_horizon=_ANNUAL_HOURS,
        timesteps_used_horizon=_ANNUAL_HOURS,
        timesteps_total=_ANNUAL_HOURS,
    )
    weather = Weather(timer, use_TRY=True)
    return Environment(timer, weather, Prices())


def _align(arr: np.ndarray) -> np.ndarray:
    """Force a 1-D array to exactly _ANNUAL_HOURS (truncate or edge-pad)."""
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size >= _ANNUAL_HOURS:
        return arr[:_ANNUAL_HOURS]
    return np.pad(arr, (0, _ANNUAL_HOURS - arr.size), mode="edge")


def _generate_pool_pycity(
    *,
    n_household_sims: int,
    persons_choices: list[int],
    pv_ref_kwp: float,
    pv_tilt_deg: float,
    pv_orientation: str,
    ev_home_requires_all: bool,
    seed: int,
    v: int,  # cache-version marker, unused at runtime
) -> dict[str, np.ndarray]:
    """Generate the annual pool of device profiles using GridCreator/pyCity."""
    import sys

    gc_path = str(Path(__file__).resolve().parents[3] / "vendor" / "GridCreator")
    if gc_path not in sys.path:
        sys.path.insert(0, gc_path)
    import creating_demand_and_load as dl  # noqa: E402  (pyCity generator fns)

    import pandas as pd

    env = _make_pycity_environment()
    index = pd.date_range("2023-01-01", periods=_ANNUAL_HOURS, freq="h")
    rng = np.random.default_rng(seed)

    base_load_sims = np.zeros((n_household_sims, _ANNUAL_HOURS))
    ev_avail_sims = np.zeros((n_household_sims, _ANNUAL_HOURS))
    for i in range(n_household_sims):
        persons = int(rng.choice(persons_choices))
        power_mw, occupancy = dl.create_appartment(env=env, people=persons, index=index)
        base_load_sims[i] = _align(np.asarray(power_mw) * 1e3)  # MW → kW

        occ_profile = np.rint(occupancy.get_occ_profile_in_curr_timestep()).astype(int)
        occ_profile = _align(occ_profile)
        threshold = occ_profile.max() if ev_home_requires_all else 1
        ev_avail_sims[i] = (occ_profile >= threshold).astype(float)

    hp_annual = _align(np.asarray(dl.create_hp(index=index, env=env)) * 1e3)  # MW → kW
    pv_annual = _align(
        np.asarray(dl.create_pv(env=env, peakpower=pv_ref_kwp, index=index,
                                beta=pv_tilt_deg, gamma=pv_orientation)) * 1e3  # MW → kW
    )
    pv_annual_per_kwp = pv_annual / pv_ref_kwp
    temp_annual = _align(np.asarray(env.weather.t_ambient))

    return dict(
        base_load_sims=base_load_sims,
        ev_avail_sims=ev_avail_sims,
        hp_annual=hp_annual,
        pv_annual_per_kwp=pv_annual_per_kwp,
        temp_annual=temp_annual,
    )
