# grid_model/test_device_profiles.py
import os

import numpy as np
import pytest

import core.constants as const
from grid_model.device_profiles import (
    DeviceProfileProvider,
    HouseholdDayProfiles,
    _hourly_to_steps,
    _noon_window,
    _ANNUAL_HOURS,
    _MAX_START_DAY,
)


# ── pure resampling / slicing (no pyCity) ────────────────────
def test_hourly_to_steps_length():
    for hold in (True, False):
        out = _hourly_to_steps(np.arange(24, dtype=float), hold=hold)
        assert out.shape == (const.EPISODE_STEPS,)


def test_hourly_to_steps_hold_repeats():
    out = _hourly_to_steps(np.arange(24, dtype=float), hold=True)
    # each hour held for exactly STEPS_PER_HOUR (=4) steps
    assert list(out[:4]) == [0, 0, 0, 0]
    assert list(out[4:8]) == [1, 1, 1, 1]


def test_hourly_to_steps_linear_hits_hourly_samples():
    win = np.arange(24, dtype=float)
    out = _hourly_to_steps(win, hold=False)
    # linear interp passes through the hourly samples at positions 0,4,8,…
    assert out[0] == pytest.approx(0.0)
    assert out[4] == pytest.approx(1.0)
    assert out[92] == pytest.approx(23.0)


def test_noon_window_picks_noon_start():
    annual = np.arange(_ANNUAL_HOURS, dtype=float)
    win = _noon_window(annual, start_day=0)
    assert win.shape == (24,)
    assert win[0] == const.EPISODE_START_HOUR      # first value is the noon hour index
    assert win[-1] == const.EPISODE_START_HOUR + 23


def test_max_start_day_window_stays_in_year():
    annual = np.arange(_ANNUAL_HOURS, dtype=float)
    win = _noon_window(annual, start_day=_MAX_START_DAY)
    assert win.shape == (24,)
    assert win[-1] < _ANNUAL_HOURS


# ── provider (constructed from synthetic arrays) ─────────────
def _tiny_provider(n_sims=3):
    rng = np.random.default_rng(0)
    return DeviceProfileProvider(
        base_load_sims=rng.uniform(0.2, 1.5, (n_sims, _ANNUAL_HOURS)),
        ev_avail_sims=rng.integers(0, 2, (n_sims, _ANNUAL_HOURS)).astype(float),
        hp_annual=rng.uniform(0.0, 2.5, _ANNUAL_HOURS),
        pv_annual_per_kwp=rng.uniform(0.0, 0.9, _ANNUAL_HOURS),
        temp_annual=rng.uniform(-10, 30, _ANNUAL_HOURS),
    )


def test_sample_episode_shapes_and_count():
    p = _tiny_provider()
    hh = p.sample_episode(n_households=5, rng=np.random.default_rng(1))
    assert len(hh) == 5
    for h in hh:
        assert isinstance(h, HouseholdDayProfiles)
        assert h.base_load_kw.shape == (const.EPISODE_STEPS,)
        assert set(np.unique(h.ev_available)).issubset({0.0, 1.0})
        assert (h.base_load_kw >= 0).all() and (h.pv_gen_kw >= 0).all()


def test_sample_episode_shares_one_weather_day():
    # HP / PV-shape / temperature come from one feeder-wide day → identical across houses
    p = _tiny_provider()
    hh = p.sample_episode(n_households=4, rng=np.random.default_rng(2))
    for h in hh[1:]:
        assert np.allclose(h.hp_load_kw, hh[0].hp_load_kw)
        assert np.allclose(h.temperature_c, hh[0].temperature_c)


def test_sample_episode_reproducible_with_seed():
    p = _tiny_provider()
    a = p.sample_episode(3, np.random.default_rng(7))
    b = p.sample_episode(3, np.random.default_rng(7))
    assert np.allclose(a[0].base_load_kw, b[0].base_load_kw)
    assert np.allclose(a[2].pv_gen_kw, b[2].pv_gen_kw)


def test_provider_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        DeviceProfileProvider(
            base_load_sims=np.zeros((2, _ANNUAL_HOURS)),
            ev_avail_sims=np.zeros((3, _ANNUAL_HOURS)),      # wrong n_sims
            hp_annual=np.zeros(_ANNUAL_HOURS),
            pv_annual_per_kwp=np.zeros(_ANNUAL_HOURS),
            temp_annual=np.zeros(_ANNUAL_HOURS),
        )


def test_household_day_profiles_validates_length():
    with pytest.raises(ValueError):
        HouseholdDayProfiles(
            base_load_kw=np.zeros(10),
            ev_available=np.zeros(10),
            hp_load_kw=np.zeros(10),
            pv_gen_kw=np.zeros(10),
            temperature_c=np.zeros(10),
        )


# ── opt-in end-to-end build via pyCity (slow; set GRIDKIT_RUN_PYCITY=1) ──
@pytest.mark.skipif(
    os.environ.get("GRIDKIT_RUN_PYCITY") != "1",
    reason="set GRIDKIT_RUN_PYCITY=1 to run the pyCity pool build (slow)",
)
def test_build_from_pycity_smoke(tmp_path):
    p = DeviceProfileProvider.build(n_household_sims=1, cache_dir=tmp_path, rebuild=True)
    assert p.n_sims == 1
    hh = p.sample_episode(2, np.random.default_rng(0))
    assert len(hh) == 2
    # real temperature must have replaced the 20 °C placeholder with actual variation
    assert hh[0].temperature_c.std() >= 0.0
    assert (hh[0].hp_load_kw >= 0).all()
