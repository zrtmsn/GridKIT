# grid_model/ev_availability.py
# Mimics GridCreator's own EV availability model
# (vendor/GridCreator/creating_demand_and_load.py:create_e_car): "available"
# means every resident of the household is home at once, from a real
# pycity_base/richardsonpy stochastic occupancy simulation — the actual
# method GridCreator itself calls, not a reimplementation of it.
# Intended for EVs added outside the GridCreator pipeline (e.g. the stub
# network), as a middle ground between real GridCreator data and a flat
# always-connected fallback.

import random

from pycity_base.classes.demand.occupancy import Occupancy
from pycity_base.classes.environment import Environment
from pycity_base.classes.timer import Timer

import core.constants as const


def synthetic_ev_availability(household_size: int, seed: int | None = None) -> list[bool]:
    """
    Return a plugged-in/away array for one representative day (length
    EPISODE_STEPS). Deterministic for a given (household_size, seed) —
    richardsonpy's occupancy model draws from Python's global `random`
    module, so its state is saved and restored around the call rather than
    permanently reseeded as a side effect.
    """
    rng_state = random.getstate()
    try:
        if seed is not None:
            random.seed(seed)
        timer = Timer(time_discretization=const.TIMESTEP_MINUTES * 60)
        environment = Environment(timer, None, None)  # weather/prices unused by Occupancy
        # nb_days=1: only one representative day is ever used below, and this
        # runs once per household bus without real GridCreator EV data — the
        # default 365-day profile would be pure wasted computation.
        occupancy = Occupancy(environment, number_occupants=household_size, nb_days=1)
        current_occupancy = occupancy.get_occ_profile_in_curr_timestep()
    finally:
        random.setstate(rng_state)

    max_occ = max(current_occupancy)
    return [bool(count == max_occ) for count in current_occupancy[: const.EPISODE_STEPS]]
