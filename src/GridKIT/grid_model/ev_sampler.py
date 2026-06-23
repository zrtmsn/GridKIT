# grid_model/ev_sampler.py
# ─────────────────────────────────────────────────────────────
# Per-episode sampling of heterogeneous EV state.
#
# This heterogeneity (different arrival/departure/SoC per household) is
# what breaks the lockstep degeneracy: under a shared policy, agents only
# desynchronize if their private state differs.
# ─────────────────────────────────────────────────────────────
import numpy as np

import core.constants as const
from core.models import EVState
from grid_model.profiles import hour_to_step


def sample_evs(agent_ids: list[str], rng: np.random.Generator) -> dict[str, EVState]:
    """
    Sample an EVState per agent for one episode.

    arrival/departure are drawn from the EV_*_HOUR Normals, mapped to
    episode-local steps, and clipped so that departure > arrival by at
    least EV_MIN_CONNECTED_STEPS (keeps EVState's validator happy and the
    car plugged in long enough to matter).
    """
    evs: dict[str, EVState] = {}
    for agent_id in agent_ids:
        arrival_hour = rng.normal(const.EV_ARRIVAL_HOUR_MEAN, const.EV_ARRIVAL_HOUR_STD)
        departure_hour = rng.normal(const.EV_DEPARTURE_HOUR_MEAN, const.EV_DEPARTURE_HOUR_STD)

        arrival_step = int(np.clip(
            hour_to_step(arrival_hour),
            const.EV_ARRIVAL_STEP_MIN,
            const.EV_DEPARTURE_STEP_MAX - const.EV_MIN_CONNECTED_STEPS,
        ))
        departure_step = int(np.clip(
            hour_to_step(departure_hour),
            arrival_step + const.EV_MIN_CONNECTED_STEPS,
            const.EV_DEPARTURE_STEP_MAX,
        ))

        soc = float(np.clip(
            rng.normal(const.EV_INITIAL_SOC_MEAN, const.EV_INITIAL_SOC_STD),
            0.05, 0.95,
        ))

        evs[agent_id] = EVState(
            agent_id=agent_id,
            bus_id=agent_id,
            soc=soc,
            arrival_step=arrival_step,
            departure_step=departure_step,
            is_connected=False,   # set per-step by the environment from the schedule
        )
    return evs
