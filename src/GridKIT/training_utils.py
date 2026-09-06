# training_utils.py
# Shared helper for the test_run_training* scripts only — not part of
# grid_model's own public API. GridEnv's real default is a pyCity-derived,
# occupancy-based EV availability profile (grid_model.device_profiles).

import numpy as np

from GridKIT.core import constants
from GridKIT.grid_model.device_profiles import DeviceProfileProvider, _ANNUAL_HOURS
from GridKIT.grid_model.environment import GridEnv


def force_always_connected(env: GridEnv) -> GridEnv:
    """
    Restore pre-synthetic-availability behavior: every EV connected for the
    whole episode, instead of GridEnv's real default (a pyCity occupancy-driven
    profile). Training here was built around always-available EVs; this keeps
    these scripts running the same as before that default changed.

    Must be called before the first reset() — GridEnv only builds its
    per-episode device profiles (base load, EV availability, HP, PV,
    temperature — all sampled together from one DeviceProfileProvider) lazily,
    the first time reset() runs, so there's no per-EV state to patch yet at
    this point; instead this swaps in a provider whose EV-availability array
    is always 1.0. The other exogenous series come along for the ride (bundled
    in the same provider) — kept at plausible constants, not zeroed out.
    """
    ones = np.ones(_ANNUAL_HOURS)
    env._provider = DeviceProfileProvider(
        base_load_sims=np.full((1, _ANNUAL_HOURS), constants.HOUSEHOLD_BASE_LOAD_MEAN_KW),
        ev_avail_sims=np.full((1, _ANNUAL_HOURS), 1.0),
        hp_annual=ones * 0.0,
        pv_annual_per_kwp=ones * 0.0,
        temp_annual=ones * 10.0,
    )
    return env
