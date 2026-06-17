# grid_model/bdew_h0.py
# Builds a BDEW H0 residential load profile using demandlib (oemof).
# Day type (weekday/Saturday/Sunday) and season are derived from actual
# calendar dates in the snapshot index, so multi-day and seasonal
# episodes work correctly without any code changes here.

import warnings

import pandas as pd
from demandlib.bdew import ElecSlp

# TODO: move to core/constants.py once agreed across modules
BDEW_H0_ANNUAL_KWH: float = 3500.0  # assumed annual household consumption (kWh/year)


def build_h0_profile(
    snapshots: pd.DatetimeIndex,
    annual_kwh: float = BDEW_H0_ANNUAL_KWH,
) -> pd.Series:
    """Return BDEW H0 base load in kW aligned to the given PyPSA snapshot index.

    Args:
        snapshots:   network.snapshots — DatetimeIndex with 15-min frequency.
        annual_kwh:  Assumed annual household consumption in kWh.

    Returns:
        Series of average power (kW) per slot, indexed by snapshots.
    """
    year = snapshots[0].year
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        e_slp = ElecSlp(year, holidays={})
    # get_scaled_power_profiles returns kW when annual_kwh is in kWh (conversion_factor=4)
    profile = e_slp.get_scaled_power_profiles({"h0": annual_kwh})
    return profile["h0"].reindex(snapshots)
