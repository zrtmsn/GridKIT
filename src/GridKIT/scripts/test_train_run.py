from scripts.train_run import household_devices_from_config


def _config(ev_ids, hp_ids):
    return {"resolved": {"ev_bus_ids": ev_ids, "heat_pump_bus_ids": hp_ids, "load_scaling_by_bus": {}}}


def test_household_devices_from_config_maps_ev_and_hp():
    config = _config(ev_ids=["h0", "h2"], hp_ids=["h1"])
    layout = household_devices_from_config(config, ["h0", "h1", "h2"])

    assert layout["h0"].ev is True and layout["h0"].heat_pump is False
    assert layout["h1"].ev is False and layout["h1"].heat_pump is True
    assert layout["h2"].ev is True and layout["h2"].heat_pump is False


def test_household_devices_from_config_battery_and_pv_always_off():
    config = _config(ev_ids=["h0"], hp_ids=["h0"])
    layout = household_devices_from_config(config, ["h0"])
    assert layout["h0"].battery is False
    assert layout["h0"].pv is False


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
