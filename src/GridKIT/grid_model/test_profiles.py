# grid_model/test_profiles.py
import numpy as np

import core.constants as const
from grid_model.profiles import hour_to_step, price_profile, base_load_profile


def test_hour_to_step_noon_start():
    assert hour_to_step(const.EPISODE_START_HOUR) == 0
    assert hour_to_step(18) == 24    # evening arrival
    assert hour_to_step(7) == 76     # morning departure (next day, no wrap)


def test_price_profile_length_and_positive():
    p = price_profile(np.random.default_rng(0), "medium")
    assert len(p) == const.EPISODE_STEPS
    assert (p > 0).all()


def test_price_cheapest_overnight_inside_connected_window():
    # the synchronization window must fall while EVs are plugged in (≈ step 24..76)
    p = price_profile(np.random.default_rng(0), "medium")
    assert 40 <= int(p.argmin()) <= 76


def test_price_scenarios_ordered():
    rng = lambda: np.random.default_rng(0)
    lo = price_profile(rng(), "low").mean()
    md = price_profile(rng(), "medium").mean()
    hi = price_profile(rng(), "high").mean()
    assert lo < md < hi


def test_base_load_profile_in_range():
    b = base_load_profile(np.random.default_rng(1))
    assert len(b) == const.EPISODE_STEPS
    assert b.min() >= 0
    # bounded by mean..peak scaled by the max load multiplier
    assert b.max() <= const.HOUSEHOLD_BASE_LOAD_PEAK_KW * const.LOAD_MULTIPLIER_MAX + 1e-9


def test_profiles_reproducible_with_seed():
    a = price_profile(np.random.default_rng(7), "medium")
    b = price_profile(np.random.default_rng(7), "medium")
    assert np.allclose(a, b)
