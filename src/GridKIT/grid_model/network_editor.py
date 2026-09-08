# grid_model/network_editor.py
# Mutation helpers for manually editing a GridNetwork — the "configure the
# network manually (add loads, etc.)" step of the map_ui workflow.
#
# Each function takes a GridNetwork and returns a NEW one (model_copy) rather
# than mutating the argument in place, since a UI doing interactive editing
# will typically want to keep the previous version around (undo, before/after
# comparison) rather than have it silently change underneath it.

import numpy as np

import core.constants as const
from core.models import BusModel, GridNetwork, LineModel
from grid_model.profiles import base_load_profile, hour_to_step


def add_bus(network: GridNetwork, bus: BusModel, line: LineModel | None = None) -> GridNetwork:
    """
    Add a new bus, optionally connected into the existing topology via a new
    line (line.from_bus or line.to_bus must equal bus.bus_id; the other end
    must already exist in the network).
    """
    if line is not None:
        if bus.bus_id not in (line.from_bus, line.to_bus):
            raise ValueError("line must connect the new bus (line.from_bus or line.to_bus must equal bus.bus_id)")
        other_end = line.to_bus if line.from_bus == bus.bus_id else line.from_bus
        existing_ids = {b.bus_id for b in network.buses}
        if other_end not in existing_ids:
            raise ValueError(f"line's other end {other_end!r} does not exist in the network")

    buses = network.buses + [bus]
    lines = network.lines + [line] if line is not None else network.lines
    return network.model_copy(update={"buses": buses, "lines": lines})


def remove_bus(network: GridNetwork, bus_id: str) -> GridNetwork:
    """Remove a bus and any lines/household/EV data referencing it."""
    return network.model_copy(update={
        "buses": [b for b in network.buses if b.bus_id != bus_id],
        "lines": [ln for ln in network.lines if bus_id not in (ln.from_bus, ln.to_bus)],
        "household_bus_ids": [b for b in network.household_bus_ids if b != bus_id],
        "household_load_profile_kw": {k: v for k, v in network.household_load_profile_kw.items() if k != bus_id},
        "ev_availability": {k: v for k, v in network.ev_availability.items() if k != bus_id},
        "household_devices": {k: v for k, v in network.household_devices.items() if k != bus_id},
    })


def add_household(network: GridNetwork, bus_id: str, seed: int | None = None) -> GridNetwork:
    """
    Mark an existing bus as a household and give it a default load profile —
    the same synthetic BDEW-H0-like generator (grid_model.profiles.base_load_profile)
    GridEnv's DeviceProfileProvider falls back to for buses without real
    GridCreator/pyCity data.
    """
    existing_ids = {b.bus_id for b in network.buses}
    if bus_id not in existing_ids:
        raise ValueError(f"bus {bus_id!r} does not exist in the network — call add_bus() first")

    profile = base_load_profile(np.random.default_rng(seed)).tolist()

    household_bus_ids = network.household_bus_ids if bus_id in network.household_bus_ids \
        else network.household_bus_ids + [bus_id]
    profiles = {**network.household_load_profile_kw, bus_id: profile}
    return network.model_copy(update={"household_bus_ids": household_bus_ids, "household_load_profile_kw": profiles})


def _synthetic_ev_window(rng: np.random.Generator) -> list[bool]:
    """
    Synthetic single-EV connection window: sampled the same way GridEnv's
    heterogeneous EVs are (evening arrival, next-morning departure — no wrap,
    since the episode starts at noon), but for one bus at a time, since this
    tool edits a network one household at a time.
    """
    arrival_h = rng.normal(const.EV_ARRIVAL_HOUR_MEAN, const.EV_ARRIVAL_HOUR_STD)
    departure_h = rng.normal(const.EV_DEPARTURE_HOUR_MEAN, const.EV_DEPARTURE_HOUR_STD)
    arrival = int(np.clip(hour_to_step(arrival_h), const.EV_ARRIVAL_STEP_MIN, const.EPISODE_STEPS - 1))
    departure = int(np.clip(hour_to_step(departure_h), arrival + const.EV_MIN_CONNECTED_STEPS, const.EV_DEPARTURE_STEP_MAX))
    return [arrival <= s <= departure for s in range(const.EPISODE_STEPS)]


def add_ev(network: GridNetwork, bus_id: str, seed: int | None = None) -> GridNetwork:
    """Attach a synthetic EV connection-window availability profile to a household bus."""
    if bus_id not in network.household_bus_ids:
        raise ValueError(f"bus {bus_id!r} is not a household bus — call add_household() first")

    availability = _synthetic_ev_window(np.random.default_rng(seed))
    return network.model_copy(update={"ev_availability": {**network.ev_availability, bus_id: availability}})
