# dashboard/geraete.py
# ─────────────────────────────────────────────────────────────
# Geräte & Haushalte — was EV, Batterie und Wärmepumpe tatsächlich getan haben.
#
# Reads only agreed fields:
#   summary.json    hp_comfort_*, battery_throughput_kwh_*, battery_full_cycles_*, soc_*
#   timelines.json  ev_power, battery_power, hp_power, pv_generation,
#                   house_ev_power / _soc / _available,
#                   house_battery_power / _soc / _charge_cumulative_kwh / _discharge_cumulative_kwh,
#                   house_hp_power / _soc
#
# The one derived value in the whole dashboard lives here: an indoor
# temperature approximated from the heat pump's thermal buffer, because
# "0,72" tells a reader nothing and "21,7 °C" tells them everything. It is a
# display conversion with declared assumptions, NOT a simulated temperature —
# see INDOOR_TEMP_* below.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from dashboard import theme
from dashboard.auslastung import format_hour, hours_axis
from dashboard.controls import scenario_penetration_picker
from dashboard.export import download_pair

# ── Innenraumtemperatur: Annahmen, nicht Simulation ──────────
# The environment models the heat pump as a thermal buffer (house inertia) and
# never computes a temperature. Turning the buffer level into °C therefore needs
# two anchors, chosen so the numbers mean something to a reader:
#   * at the model's own comfort floor the house sits at 20 °C — the German
#     design indoor temperature for living space (DIN EN 12831),
#   * a full buffer is 23 °C, a comfortably warm house.
# Linear between and beyond. Change these two numbers and every temperature in
# the view moves with them; nothing else depends on the mapping.
INDOOR_TEMP_AT_COMFORT_MIN_C = 20.0
INDOOR_TEMP_AT_FULL_C = 23.0
HP_COMFORT_MIN_SOC = 0.30       # mirrors core.constants.HP_COMFORT_MIN_SOC


# ══════════════════════════════════════════════════════════════
# Reine Helfer (kein Streamlit — unit-testbar)
# ══════════════════════════════════════════════════════════════
def indoor_temperature(thermal_soc: float) -> float:
    """Approximate indoor temperature (°C) from the heat pump's buffer level.

    Linear through the two anchors above: the comfort floor maps to 20 °C and a
    full buffer to 23 °C. Values outside [0, 1] are extrapolated rather than
    clipped, so an implausible input stays visible instead of being hidden.
    """
    span_c = INDOOR_TEMP_AT_FULL_C - INDOOR_TEMP_AT_COMFORT_MIN_C
    span_soc = 1.0 - HP_COMFORT_MIN_SOC
    return INDOOR_TEMP_AT_COMFORT_MIN_C + (float(thermal_soc) - HP_COMFORT_MIN_SOC) * span_c / span_soc


COMFORT_FLOOR_C = INDOOR_TEMP_AT_COMFORT_MIN_C


def device_power_frame(timeline: dict[str, Any]) -> pd.DataFrame:
    """Long-form aggregate device power per quarter hour.

    Battery power is signed (+ charging, − discharging) and stays that way: a
    battery that never moves and one that charges then discharges the same
    energy both average to zero, and only the sign shows the difference.
    """
    series = [
        ("ev", "ev_power"),
        ("battery", "battery_power"),
        ("hp", "hp_power"),
    ]
    lengths = [len(timeline.get(field) or []) for _, field in series]
    n = max(lengths, default=0)
    hours = hours_axis(n)

    rows: list[dict[str, Any]] = []
    for device, field in series:
        for i, value in enumerate(timeline.get(field) or []):
            rows.append({
                "Stunde": float(hours[i]),
                "Uhrzeit": format_hour(float(hours[i])),
                "geraet": device,
                "Gerät": theme.DEVICE_LABELS_DE[device],
                "Leistung": float(value),
            })
    return pd.DataFrame(rows, columns=["Stunde", "Uhrzeit", "geraet", "Gerät", "Leistung"])


def household_frame(timeline: dict[str, Any]) -> pd.DataFrame:
    """One representative household: power, state of charge, indoor temperature.

    `ev_available` is carried through as a boolean so the EV chart can shade the
    window the car was actually plugged in — an EV at a flat SoC is a completely
    different story depending on whether it was home and idle or simply away.
    """
    ev_power = timeline.get("house_ev_power") or []
    n = len(ev_power) or len(timeline.get("house_hp_soc") or [])
    hours = hours_axis(n)

    def col(name: str) -> list[float]:
        values = timeline.get(name) or []
        return [float(v) for v in values] + [float("nan")] * (n - len(values))

    hp_soc = col("house_hp_soc")
    frame = pd.DataFrame({
        "Stunde": hours[:n],
        "Uhrzeit": [format_hour(float(h)) for h in hours[:n]],
        "EV (kW)": col("house_ev_power"),
        "EV-SoC": col("house_ev_soc"),
        "Batterie (kW)": col("house_battery_power"),
        "Batterie-SoC": col("house_battery_soc"),
        "WP (kW)": col("house_hp_power"),
        "WP-Puffer": hp_soc,
        "Innentemperatur": [indoor_temperature(v) for v in hp_soc],
    })
    frame["EV angeschlossen"] = [bool(v) for v in
                                 (col("house_ev_available") if timeline.get("house_ev_available") else [0.0] * n)]
    return frame


def battery_cycle_frame(timeline: dict[str, Any]) -> pd.DataFrame:
    """Cumulative charged/discharged energy and the resulting equivalent cycles."""
    charge = [float(v) for v in (timeline.get("house_battery_charge_cumulative_kwh") or [])]
    discharge = [float(v) for v in (timeline.get("house_battery_discharge_cumulative_kwh") or [])]
    n = max(len(charge), len(discharge))
    if n == 0:
        return pd.DataFrame(columns=["Stunde", "Uhrzeit", "Richtung", "Energie"])

    hours = hours_axis(n)
    rows: list[dict[str, Any]] = []
    for label, values in (("geladen", charge), ("entladen", discharge)):
        for i, value in enumerate(values):
            rows.append({
                "Stunde": float(hours[i]),
                "Uhrzeit": format_hour(float(hours[i])),
                "Richtung": label,
                "Energie": value,
            })
    return pd.DataFrame(rows)


def comfort_breaches(timeline: dict[str, Any]) -> int:
    """Quarter hours the thermal buffer spent below the model's comfort floor."""
    return sum(1 for v in (timeline.get("house_hp_soc") or []) if float(v) < HP_COMFORT_MIN_SOC)


def device_totals(timeline: dict[str, Any]) -> dict[str, float]:
    """Energy per device over the episode (kWh), battery split by direction."""
    step_h = 0.25
    ev = sum(float(v) for v in (timeline.get("ev_power") or [])) * step_h
    hp = sum(float(v) for v in (timeline.get("hp_power") or [])) * step_h
    battery = [float(v) for v in (timeline.get("battery_power") or [])]
    pv = sum(abs(float(v)) for v in (timeline.get("pv_generation") or [])) * step_h
    return {
        "ev_kwh": ev,
        "hp_kwh": hp,
        "battery_charge_kwh": sum(v for v in battery if v > 0) * step_h,
        "battery_discharge_kwh": -sum(v for v in battery if v < 0) * step_h,
        "pv_kwh": pv,
    }


def scenario_devices(summary: list[dict[str, Any]], penetration: float) -> pd.DataFrame:
    """Per-scenario device figures at one penetration, for side-by-side reading."""
    rows = []
    for record in summary:
        if abs(float(record.get("penetration", -1)) - penetration) > 1e-9:
            continue
        rows.append({
            "Szenario": theme.scenario_label(record.get("scenario", "")),
            "EV-Ziel erreicht": record.get("soc_mean"),
            "WP-Komfort": record.get("hp_comfort_mean"),
            "Batterie-Durchsatz (kWh)": record.get("battery_throughput_kwh_mean"),
            "Vollzyklen": record.get("battery_full_cycles_mean"),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Streamlit-Ansicht
# ══════════════════════════════════════════════════════════════
_HOUR_AXIS = dict(
    values=[12, 15, 18, 21, 24, 27, 30, 33, 36],
    labelExpr="format(datum.value % 24, '02') + ':00'",
)


def _device_power_chart(timeline: dict[str, Any]):  # pragma: no cover (UI)
    import altair as alt

    frame = device_power_frame(timeline)
    devices = [d for d in ("ev", "battery", "hp") if d in set(frame["geraet"])]
    zero = (
        alt.Chart(pd.DataFrame({"y": [0.0]}))
        .mark_rule(color="#adb5bd", strokeWidth=1)
        .encode(y="y:Q")
    )
    lines = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Stunde:Q", title="Uhrzeit", axis=alt.Axis(**_HOUR_AXIS)),
            y=alt.Y("Leistung:Q", title="Leistung (kW)"),
            color=alt.Color(
                "Gerät:N",
                scale=alt.Scale(domain=[theme.DEVICE_LABELS_DE[d] for d in devices],
                                range=[theme.DEVICE_COLORS[d] for d in devices]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[alt.Tooltip("Uhrzeit:N"), alt.Tooltip("Gerät:N"),
                     alt.Tooltip("Leistung:Q", format=".2f", title="kW")],
        )
    )
    return (zero + lines).properties(width="container", height=300)


def _ev_chart(frame: pd.DataFrame):  # pragma: no cover (UI)
    import altair as alt

    soc = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2.4, color=theme.DEVICE_COLORS["ev"])
        .encode(
            x=alt.X("Stunde:Q", title="Uhrzeit", axis=alt.Axis(**_HOUR_AXIS)),
            y=alt.Y("EV-SoC:Q", title="Ladestand", scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("Uhrzeit:N"), alt.Tooltip("EV-SoC:Q", format=".2f")],
        )
    )
    away = frame[~frame["EV angeschlossen"]]
    if away.empty:
        return soc.properties(width="container", height=240)
    # shade the time the car was NOT plugged in — a flat SoC means something
    # different depending on whether the car was home and idle or simply gone
    shade = (
        alt.Chart(away)
        .mark_rect(opacity=0.13, color="#6c757d")
        .encode(x=alt.X("Stunde:Q"), x2="Stunde:Q")
    )
    return (shade + soc).properties(width="container", height=240)


def _battery_chart(frame: pd.DataFrame, cycles: pd.DataFrame):  # pragma: no cover (UI)
    import altair as alt

    soc = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2.4, color=theme.DEVICE_COLORS["battery"])
        .encode(
            x=alt.X("Stunde:Q", title="Uhrzeit", axis=alt.Axis(**_HOUR_AXIS)),
            y=alt.Y("Batterie-SoC:Q", title="Ladestand", scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("Uhrzeit:N"), alt.Tooltip("Batterie-SoC:Q", format=".2f")],
        )
        .properties(width="container", height=220)
    )
    if cycles.empty:
        return soc
    energy = (
        alt.Chart(cycles)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Stunde:Q", title="Uhrzeit", axis=alt.Axis(**_HOUR_AXIS)),
            y=alt.Y("Energie:Q", title="kumulierte Energie (kWh)"),
            color=alt.Color("Richtung:N",
                            scale=alt.Scale(domain=["geladen", "entladen"],
                                            range=[theme.DEVICE_COLORS["battery"], "#e76f51"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("Uhrzeit:N"), alt.Tooltip("Richtung:N"),
                     alt.Tooltip("Energie:Q", format=".1f", title="kWh")],
        )
        .properties(width="container", height=220)
    )
    return soc, energy


def _hp_chart(frame: pd.DataFrame):  # pragma: no cover (UI)
    import altair as alt

    floor = (
        alt.Chart(pd.DataFrame({"y": [COMFORT_FLOOR_C]}))
        .mark_rule(color=theme.STATUS_COLORS["kritisch"], strokeDash=[5, 4], strokeWidth=1.4)
        .encode(y=alt.Y("y:Q", title="Innentemperatur (°C)"))
    )
    line = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2.4, color=theme.DEVICE_COLORS["hp"])
        .encode(
            x=alt.X("Stunde:Q", title="Uhrzeit", axis=alt.Axis(**_HOUR_AXIS)),
            y=alt.Y("Innentemperatur:Q", title="Innentemperatur (°C)",
                    scale=alt.Scale(zero=False, nice=True)),
            tooltip=[alt.Tooltip("Uhrzeit:N"),
                     alt.Tooltip("Innentemperatur:Q", format=".1f", title="°C"),
                     alt.Tooltip("WP-Puffer:Q", format=".2f", title="Puffer")],
        )
    )
    return (floor + line).properties(width="container", height=240)


def render_geraete(summary: list[dict[str, Any]], timelines: list[dict[str, Any]],
                   key: str = "geraete") -> None:  # pragma: no cover (UI)
    """Der Geräte-Reiter: Leistung je Gerätetyp und ein Haushalt im Detail."""
    import streamlit as st

    from dashboard.auslastung import select_timeline

    if not timelines:
        st.info("Keine Zeitreihen vorhanden. Zuerst ein Experiment ausführen.")
        return

    scenario, penetration = scenario_penetration_picker(timelines, key)
    if scenario is None:
        st.info("Keine auswertbaren Zeitreihen vorhanden.")
        return

    timeline = select_timeline(timelines, scenario, penetration)
    if timeline is None:
        st.warning("Für diese Kombination liegt keine Zeitreihe vor.")
        return

    totals = device_totals(timeline)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EV geladen", f"{totals['ev_kwh']:.1f} kWh".replace(".", ","))
    c2.metric("Wärmepumpe", f"{totals['hp_kwh']:.1f} kWh".replace(".", ","))
    c3.metric("Batterie geladen", f"{totals['battery_charge_kwh']:.1f} kWh".replace(".", ","),
              delta=f"−{totals['battery_discharge_kwh']:.1f} kWh entladen".replace(".", ","),
              delta_color="off")
    c4.metric("PV erzeugt", f"{totals['pv_kwh']:.1f} kWh".replace(".", ","))

    st.subheader("Leistung je Gerätetyp")
    st.caption(
        "Alle Haushalte zusammen. Die Batterie ist **vorzeichenbehaftet**: positiv lädt, "
        "negativ speist ins Haus zurück. Eine Batterie, die sich nie bewegt, und eine, "
        "die lädt und wieder entlädt, ergeben im Mittel beide null."
    )
    st.altair_chart(_device_power_chart(timeline), width="stretch")
    download_pair(device_power_frame(timeline), "Geräteleistung", f"{key}_leistung")

    # ── Szenarienvergleich der Gerätekennzahlen ───────────────
    table = scenario_devices(summary or [], penetration)
    if not table.empty and table.drop(columns=["Szenario"]).notna().any().any():
        st.subheader(f"Gerätekennzahlen bei {penetration:.0%} Ausstattungsgrad")
        st.dataframe(
            table.style.format({
                "EV-Ziel erreicht": "{:.0%}", "WP-Komfort": "{:.0%}",
                "Batterie-Durchsatz (kWh)": "{:.1f}", "Vollzyklen": "{:.2f}",
            }, na_rep=theme.NO_VALUE),
            width="stretch", hide_index=True,
        )
        download_pair(table, "Gerätekennzahlen", f"{key}_kennzahlen")

    # ── Ein Haushalt im Detail ────────────────────────────────
    st.subheader("Ein repräsentativer Haushalt")
    frame = household_frame(timeline)
    if frame.empty:
        st.info("Für diesen Lauf wurden keine Haushaltsdetails aufgezeichnet.")
        return
    download_pair(frame, "Haushalt", f"{key}_haushalt",
                  label="Alle Reihen dieses Haushalts")

    ev_tab, batt_tab, hp_tab = st.tabs(["EV", "Batterie", "Wärmepumpe"])

    with ev_tab:
        st.caption(
            "Grau hinterlegt: das Auto war nicht angeschlossen. Ein flacher Ladestand "
            "heißt in dieser Zeit „weg“, nicht „nicht geladen“."
        )
        st.altair_chart(_ev_chart(frame), width="stretch")

    with batt_tab:
        cycles = battery_cycle_frame(timeline)
        charts = _battery_chart(frame, cycles)
        if isinstance(charts, tuple):
            soc_chart, energy_chart = charts
            st.altair_chart(soc_chart, width="stretch")
            st.caption(
                "Kumulierte Energie beider Richtungen, die Grundlage der äquivalenten "
                "Vollzyklen, die das Szenario oben ausweist."
            )
            st.altair_chart(energy_chart, width="stretch")
        else:
            st.altair_chart(charts, width="stretch")

    with hp_tab:
        breaches = comfort_breaches(timeline)
        if breaches:
            st.warning(
                f"{breaches} Viertelstunden unter der Komfortgrenze "
                f"({COMFORT_FLOOR_C:.0f} °C)."
            )
        else:
            st.success(f"Durchgehend über der Komfortgrenze ({COMFORT_FLOOR_C:.0f} °C).")
        st.caption(
            "**Näherung.** Das Modell rechnet mit einem thermischen Puffer, nicht mit "
            f"Temperatur. Hier ist die Komfortgrenze des Modells auf {COMFORT_FLOOR_C:.0f} °C "
            f"gelegt und ein voller Puffer auf {INDOOR_TEMP_AT_FULL_C:.0f} °C, dazwischen "
            "linear. Die Kurvenform ist echt, die absoluten Werte hängen an diesen zwei Annahmen."
        )
        st.altair_chart(_hp_chart(frame), width="stretch")
