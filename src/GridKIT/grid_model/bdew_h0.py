# grid_model/bdew_h0.py
# Representative BDEW H0 residential load profile (96 × 15-min slots, one day).
# Shape derived from the official BDEW standard load profile for households
# (working day, transition season).  Values are pre-scaled to a household that
# consumes BDEW_H0_ANNUAL_KWH per year; multiply by a per-episode load multiplier
# to vary grid stress across training episodes.
#
# demandlib (oemof) was considered but requires calendar-date-anchored pandas
# DatetimeIndex, which doesn't fit the abstract 0–95 episode timestep model.

from core.constants import EPISODE_STEPS

# TODO: move to core/constants.py once the profile shape is agreed across modules
BDEW_H0_ANNUAL_KWH: float = 3500.0  # assumed annual household consumption (kWh/year)

# Instantaneous power in kW for each 15-min slot (00:00 … 23:45).
# sum × 0.25 h ≈ 9.7 kWh/day  →  annual ≈ 3540 kWh (≈ 1 % above BDEW_H0_ANNUAL_KWH,
# normal rounding in published profile tables).
H0_PROFILE_KW: tuple[float, ...] = (
    # 00:00 – 05:45  (night)
    0.22, 0.21, 0.20, 0.19, 0.19, 0.18, 0.18, 0.17,
    0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17,
    0.17, 0.17, 0.18, 0.19, 0.21, 0.23, 0.27, 0.32,
    # 06:00 – 09:45  (morning ramp)
    0.38, 0.44, 0.49, 0.53, 0.56, 0.58, 0.59, 0.58,
    0.57, 0.54, 0.51, 0.48, 0.45, 0.43, 0.41, 0.40,
    # 10:00 – 16:45  (midday plateau + pre-evening rise)
    0.39, 0.38, 0.38, 0.37, 0.37, 0.37, 0.37, 0.37,
    0.38, 0.39, 0.40, 0.41, 0.42, 0.42, 0.42, 0.42,
    0.41, 0.41, 0.41, 0.41, 0.42, 0.43, 0.44, 0.46,
    0.48, 0.51, 0.54, 0.57,
    # 17:00 – 21:45  (evening peak)
    0.61, 0.64, 0.67, 0.70, 0.72, 0.73, 0.73, 0.72,
    0.71, 0.69, 0.67, 0.64, 0.61, 0.57, 0.54, 0.50,
    0.46, 0.43, 0.40, 0.37,
    # 22:00 – 23:45  (late night)
    0.35, 0.33, 0.31, 0.30, 0.28, 0.27, 0.25, 0.23,
)

assert len(H0_PROFILE_KW) == EPISODE_STEPS, (
    f"H0_PROFILE_KW must have {EPISODE_STEPS} entries, got {len(H0_PROFILE_KW)}"
)


def get_base_load_kw(step: int, multiplier: float = 1.0) -> float:
    """Return inflexible household base load in kW for the given timestep.

    Args:
        step:       Timestep index (0–95); wraps via modulo.
        multiplier: Per-episode scaling factor sampled from
                    [LOAD_MULTIPLIER_MIN, LOAD_MULTIPLIER_MAX] at reset().
    """
    return H0_PROFILE_KW[step % EPISODE_STEPS] * multiplier
