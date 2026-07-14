# training_utils.py
# Shared helper for the test_run_training* scripts only — not part of
# grid_model's own public API, since GridEnv's real default is the
# synthetic, occupancy-based EV availability (grid_model.ev_availability).

from GridKIT.core import constants
from GridKIT.grid_model.environment import GridEnv


def force_always_connected(env: GridEnv) -> GridEnv:
    """
    Restore pre-synthetic-availability behavior: every EV connected for the
    whole episode (steps 0-23), instead of GridEnv's real default (a
    synthetic, occupancy-based profile for buses without real GridCreator
    data). Training here was built around always-available EVs; this keeps
    these scripts running the same as before that default changed.
    """
    for ev in env._evs.values():
        ev.availability = [True] * constants.EPISODE_STEPS
    return env
