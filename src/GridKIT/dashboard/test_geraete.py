# dashboard/test_geraete.py
import math

import core.constants as const
from dashboard.geraete import (
    COMFORT_FLOOR_C,
    HP_COMFORT_MIN_SOC,
    INDOOR_TEMP_AT_FULL_C,
    battery_cycle_frame,
    comfort_breaches,
    device_power_frame,
    device_totals,
    household_frame,
    indoor_temperature,
    scenario_devices,
)


def _timeline(n=4, **extra):
    base = {
        "scenario": "1: flat / immediate",
        "penetration": 0.6,
        "ev_power": [1.0] * n,
        "battery_power": [2.0] * n,
        "hp_power": [3.0] * n,
        "pv_generation": [0.0] * n,
        "house_ev_power": [1.0] * n,
        "house_ev_soc": [0.5] * n,
        "house_ev_available": [1.0] * n,
        "house_battery_power": [0.5] * n,
        "house_battery_soc": [0.6] * n,
        "house_hp_power": [1.5] * n,
        "house_hp_soc": [0.7] * n,
    }
    base.update(extra)
    return base


# ── Innenraumtemperatur ──────────────────────────────────────
def test_comfort_floor_soc_maps_to_the_comfort_temperature():
    assert indoor_temperature(HP_COMFORT_MIN_SOC) == COMFORT_FLOOR_C


def test_full_buffer_maps_to_the_upper_anchor():
    assert indoor_temperature(1.0) == INDOOR_TEMP_AT_FULL_C


def test_temperature_rises_with_the_buffer():
    assert indoor_temperature(0.4) < indoor_temperature(0.8)


def test_below_the_floor_reads_below_the_comfort_temperature():
    # the whole point of the conversion: a comfort breach must look like one
    assert indoor_temperature(0.1) < COMFORT_FLOOR_C


def test_mapping_is_linear():
    a = indoor_temperature(0.4) - indoor_temperature(0.3)
    b = indoor_temperature(0.8) - indoor_temperature(0.7)
    assert math.isclose(a, b, rel_tol=1e-9)


def test_dashboard_floor_matches_the_model_constant():
    # if the environment's comfort floor moves, the temperature anchor must follow
    assert HP_COMFORT_MIN_SOC == const.HP_COMFORT_MIN_SOC


# ── Geräteleistung ───────────────────────────────────────────
def test_device_power_frame_has_a_row_per_device_and_step():
    frame = device_power_frame(_timeline(n=3))
    assert len(frame) == 9
    assert set(frame["geraet"]) == {"ev", "battery", "hp"}


def test_device_power_frame_keeps_the_battery_sign():
    frame = device_power_frame(_timeline(n=2, battery_power=[2.0, -2.0]))
    battery = frame[frame["geraet"] == "battery"]["Leistung"].tolist()
    assert battery == [2.0, -2.0]


def test_device_power_frame_uses_german_labels():
    frame = device_power_frame(_timeline(n=1))
    assert "Wärmepumpe" in set(frame["Gerät"])


# ── Energiesummen ────────────────────────────────────────────
def test_device_totals_convert_power_to_energy_over_quarter_hours():
    # 4 steps at 1 kW = 1 kWh
    assert device_totals(_timeline(n=4))["ev_kwh"] == 1.0


def test_device_totals_split_battery_by_direction():
    totals = device_totals(_timeline(n=4, battery_power=[4.0, 4.0, -4.0, -4.0]))
    assert totals["battery_charge_kwh"] == 2.0
    assert totals["battery_discharge_kwh"] == 2.0


def test_device_totals_treat_pv_as_magnitude():
    # pv is exported as generation; sign convention should not flip the total
    totals = device_totals(_timeline(n=4, pv_generation=[-2.0] * 4))
    assert totals["pv_kwh"] == 2.0


# ── Haushaltsdetail ──────────────────────────────────────────
def test_household_frame_adds_a_temperature_column():
    frame = household_frame(_timeline(n=3))
    assert "Innentemperatur" in frame.columns
    assert frame["Innentemperatur"].iloc[0] == indoor_temperature(0.7)


def test_household_frame_marks_when_the_ev_was_plugged_in():
    frame = household_frame(_timeline(n=3, house_ev_available=[1.0, 0.0, 1.0]))
    assert frame["EV angeschlossen"].tolist() == [True, False, True]


def test_household_frame_pads_a_short_series_instead_of_failing():
    # an older run may not carry every per-household series
    frame = household_frame(_timeline(n=4, house_battery_soc=[0.6, 0.6]))
    assert len(frame) == 4
    assert math.isnan(frame["Batterie-SoC"].iloc[3])


# ── Batteriezyklen ───────────────────────────────────────────
def test_battery_cycle_frame_carries_both_directions():
    tl = _timeline(n=3,
                   house_battery_charge_cumulative_kwh=[0.0, 1.0, 2.0],
                   house_battery_discharge_cumulative_kwh=[0.0, 0.5, 1.0])
    frame = battery_cycle_frame(tl)
    assert set(frame["Richtung"]) == {"geladen", "entladen"}
    assert len(frame) == 6


def test_battery_cycle_frame_without_the_series_is_empty():
    assert battery_cycle_frame(_timeline(n=3)).empty


# ── Komfort ──────────────────────────────────────────────────
def test_comfort_breaches_counts_steps_below_the_floor():
    tl = _timeline(n=5, house_hp_soc=[0.5, 0.2, 0.1, 0.4, 0.29])
    assert comfort_breaches(tl) == 3


def test_no_breaches_when_the_buffer_stays_up():
    assert comfort_breaches(_timeline(n=4)) == 0


# ── Szenariotabelle ──────────────────────────────────────────
def test_scenario_devices_filters_to_one_penetration():
    summary = [
        {"penetration": 0.2, "scenario": "1: flat / immediate", "soc_mean": 1.0},
        {"penetration": 0.6, "scenario": "1: flat / immediate", "soc_mean": 0.8},
    ]
    table = scenario_devices(summary, 0.6)
    assert len(table) == 1
    assert table["EV-Ziel erreicht"].iloc[0] == 0.8


def test_scenario_devices_tolerates_a_run_without_the_battery_fields():
    # older summary.json predates the battery columns; the tab must still render
    summary = [{"penetration": 0.6, "scenario": "1: flat / immediate", "soc_mean": 0.8}]
    table = scenario_devices(summary, 0.6)
    assert table["Vollzyklen"].isna().all()
