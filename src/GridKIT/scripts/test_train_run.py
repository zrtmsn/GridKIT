from scripts.train_run import configured_share, household_devices_from_config


def _config(ev_ids, hp_ids):
    return {"resolved": {"ev_bus_ids": ev_ids, "heat_pump_bus_ids": hp_ids, "load_scaling_by_bus": {}}}


def test_household_devices_from_config_maps_ev_and_hp():
    config = _config(ev_ids=["h0", "h2"], hp_ids=["h1"])
    layout = household_devices_from_config(config, ["h0", "h1", "h2"])

    assert layout["h0"].ev is True and layout["h0"].heat_pump is False
    assert layout["h1"].ev is False and layout["h1"].heat_pump is True
    assert layout["h2"].ev is True and layout["h2"].heat_pump is False


def test_household_devices_from_config_battery_and_pv_default_off_when_absent():
    # old saved configs (or networks with no GridCreator device data) have no
    # battery_bus_ids/pv_bus_ids keys at all — must default to off, not KeyError.
    config = _config(ev_ids=["h0"], hp_ids=["h0"])
    layout = household_devices_from_config(config, ["h0"])
    assert layout["h0"].battery is False
    assert layout["h0"].pv is False
    assert layout["h0"].battery_kwh is None
    assert layout["h0"].pv_kwp is None


def test_household_devices_from_config_maps_battery_and_pv():
    config = {
        "resolved": {
            "ev_bus_ids": [], "heat_pump_bus_ids": [],
            "battery_bus_ids": ["h0"], "pv_bus_ids": ["h0", "h1"],
            "pv_kwp_by_bus": {"h0": 6.5, "h1": 4.0},
            "battery_kwh_by_bus": {"h0": 12.0},
        }
    }
    layout = household_devices_from_config(config, ["h0", "h1"])

    assert layout["h0"].battery is True and layout["h0"].battery_kwh == 12.0
    assert layout["h0"].pv is True and layout["h0"].pv_kwp == 6.5
    assert layout["h1"].battery is False and layout["h1"].battery_kwh is None
    assert layout["h1"].pv is True and layout["h1"].pv_kwp == 4.0


def test_household_devices_from_config_covers_every_household_bus_id():
    config = _config(ev_ids=[], hp_ids=[])
    layout = household_devices_from_config(config, ["h0", "h1"])
    assert set(layout) == {"h0", "h1"}
    assert all(not d.ev and not d.heat_pump for d in layout.values())


def test_household_devices_from_config_ignores_unknown_resolved_ids():
    # a bus_id present in "resolved" but not in household_bus_ids (e.g. stale
    # config from a previously-built network) must not produce a phantom entry
    config = _config(ev_ids=["ghost"], hp_ids=[])
    layout = household_devices_from_config(config, ["h0"])
    assert set(layout) == {"h0"}
    assert layout["h0"].ev is False


# ── Ausstattungsgrad eines Karten-Laufs ──────────────────────
def test_configured_share_counts_households_with_a_controllable_device():
    layout = household_devices_from_config(_config(ev_ids=["h0"], hp_ids=["h1"]),
                                           ["h0", "h1", "h2", "h3"])
    assert configured_share(layout) == 0.5


def test_configured_share_ignores_pv_only_households():
    # PV is exogenous, not controllable, so a PV-only household is not part of
    # what the Ausstattungsgrad describes
    config = {"resolved": {"ev_bus_ids": [], "heat_pump_bus_ids": [],
                           "pv_bus_ids": ["h0", "h1"]}}
    layout = household_devices_from_config(config, ["h0", "h1"])
    assert configured_share(layout) == 0.0


def test_configured_share_of_a_fully_equipped_grid_is_one():
    layout = household_devices_from_config(_config(ev_ids=["h0", "h1"], hp_ids=[]),
                                           ["h0", "h1"])
    assert configured_share(layout) == 1.0


def test_configured_share_of_an_empty_layout_is_zero():
    # a network with no households at all must not divide by zero
    assert configured_share({}) == 0.0
