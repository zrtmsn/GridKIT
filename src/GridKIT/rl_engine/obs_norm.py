# rl_engine/obs_norm.py
"""
Single source of truth for mapping a core Observation → the normalized [0,1]
vector the policy network consumes. Shared by the RLlib wrapper (training) and
the RLlibPolicyAdapter (evaluation) so train-time and eval-time inputs match.
"""
import numpy as np

from GridKIT.core import constants as const
from GridKIT.core.models import Observation


def normalize_observation(obs: Observation) -> np.ndarray:
    """Map the 7 raw Observation fields into [0,1], order matching OBS_DIM."""
    soc, urgency, price, base_load, temp, voltage, curtail = obs.to_array()

    def unit(x: float, lo: float, hi: float) -> float:
        return (x - lo) / (hi - lo)

    arr = [
        soc,
        urgency,
        price / const.PRICE_NORM_MAX_EUR_KWH,
        base_load / const.BASE_LOAD_NORM_MAX_KW,
        unit(temp, const.TEMP_MIN_C, const.TEMP_MAX_C),
        unit(voltage, const.VOLTAGE_MIN_PU, const.VOLTAGE_MAX_PU),
        curtail,
    ]
    return np.clip(np.array(arr, dtype=np.float32), 0.0, 1.0)
