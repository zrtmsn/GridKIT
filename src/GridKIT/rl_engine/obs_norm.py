# rl_engine/obs_norm.py
"""
Single source of truth for mapping a core Observation → the normalized [0,1]
vector the policy network consumes. Shared by the RLlib wrapper (training) and
the RLlibPolicyAdapter (evaluation) so train-time and eval-time inputs match.
"""
import numpy as np

from GridKIT.core import constants as const
from GridKIT.core.models import Observation


def _unit(x: float, lo: float, hi: float) -> float:
    return (x - lo) / (hi - lo)


def normalize_observation(obs: Observation) -> np.ndarray:
    """Legacy 7-dim EV-only normalization (order matches OBS_DIM). Kept for tests."""
    soc, urgency, price, base_load, temp, voltage, curtail = obs.to_array()
    arr = [
        soc,
        urgency,
        price / const.PRICE_NORM_MAX_EUR_KWH,
        base_load / const.BASE_LOAD_NORM_MAX_KW,
        _unit(temp, const.TEMP_MIN_C, const.TEMP_MAX_C),
        _unit(voltage, const.VOLTAGE_MIN_PU, const.VOLTAGE_MAX_PU),
        curtail,
    ]
    return np.clip(np.array(arr, dtype=np.float32), 0.0, 1.0)


def normalize_observation_multidevice(obs: Observation) -> np.ndarray:
    """Map the 10 raw multi-device Observation fields into [0,1]
    (order matches OBS_DIM_MULTIDEVICE / Observation.to_array_multidevice)."""
    (soc, urgency, price, feed_in, net_load,
     pv_gen, temp, voltage, curtail, tod) = obs.to_array_multidevice()
    arr = [
        soc,                                                    # own device progress (can exceed 1 → clipped)
        urgency,
        price / const.PRICE_NORM_MAX_EUR_KWH,
        feed_in / const.PRICE_NORM_MAX_EUR_KWH,
        _unit(net_load, -const.NET_LOAD_NORM_MAX_KW, const.NET_LOAD_NORM_MAX_KW),  # signed → [0,1]
        pv_gen / const.PV_GEN_NORM_MAX_KW,
        _unit(temp, const.TEMP_MIN_C, const.TEMP_MAX_C),
        _unit(voltage, const.VOLTAGE_MIN_PU, const.VOLTAGE_MAX_PU),
        curtail,
        tod,
    ]
    return np.clip(np.array(arr, dtype=np.float32), 0.0, 1.0)
