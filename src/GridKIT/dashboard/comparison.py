# dashboard/comparison.py
# ─────────────────────────────────────────────────────────────
# Scenario comparison: the four control strategies side by side.
#
# What the other tabs show for ONE strategy, this one shows for all four at
# once: the same grid situation, four policies, what does each of them cost?
#
# Deliberately NOT here: the representative household. That belongs to the
# "Geräte & Haushalte" tab, where it is broken out by EV/battery/heat pump
# along with the indoor temperature. Repeating it here would mean maintaining
# the same curve twice and showing the reader the same thing twice.
#
# Altair charts using the colours from theme.py, like the rest of the dashboard.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard import theme
from dashboard.utilization import format_hour, hours_axis, select_timeline
from dashboard.controls import penetration_picker, scenario_picker
from dashboard.export import download_pair

#: (column name in summary.json, display label, unit, "higher is better")
METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("curtailment_mean", "§14a-Eingriffe", "Viertelstunden", False),
    ("soc_mean", "EV-Ziel erreicht", "Anteil", True),
    ("hp_comfort_mean", "WP-Komfort", "Anteil", True),
    ("bill_mean", "Stromkosten", "€ / Haushalt", False),
)


# ══════════════════════════════════════════════════════════════
# Pure helpers (no Streamlit, unit-tested)
# ══════════════════════════════════════════════════════════════
def metric_frame(summary: list[dict[str, Any]], penetration: float | None = None) -> pd.DataFrame:
    """Long-form frame of the comparison metrics, one row per scenario × metric.

    Carries the standard deviation alongside each mean so the chart can draw the
    spread over the evaluation seeds: a difference smaller than the spread is
    not a difference, and a bare bar would hide that.
    """
    rows: list[dict[str, Any]] = []
    for record in summary or []:
        if penetration is not None and abs(float(record.get("penetration", -1)) - penetration) > 1e-9:
            continue
        scenario = record.get("scenario", "")
        for field, label, unit, higher_better in METRICS:
            value = record.get(field)
            if value is None:
                continue
            std = record.get(field.replace("_mean", "_std"))
            rows.append({
                "scenario": scenario,
                "Szenario": theme.scenario_label(scenario),
                "kennzahl": field,
                "Kennzahl": label,
                "Einheit": unit,
                "besser": "hoch" if higher_better else "niedrig",
                "Wert": float(value),
                "Streuung": float(std) if std is not None else 0.0,
            })
    return pd.DataFrame(rows, columns=["scenario", "Szenario", "kennzahl", "Kennzahl",
                                       "Einheit", "besser", "Wert", "Streuung"])


def episode_frame(timeline: dict[str, Any]) -> pd.DataFrame:
    """The representative day as one long-form frame: loading, price, PV, devices."""
    if not timeline:
        return pd.DataFrame(columns=["Stunde", "Uhrzeit", "reihe", "Reihe", "Wert", "Gruppe"])

    series = [
        ("transformer_loading", "Trafo-Auslastung", "Auslastung", 100.0),
        ("max_line_loading", "max. Leitungsauslastung", "Auslastung", 100.0),
        ("price", "Strompreis", "Preis", 1.0),
        ("pv_generation", "PV-Erzeugung", "Leistung", 1.0),
        ("ev_power", "EV", "Leistung", 1.0),
        ("battery_power", "Batterie", "Leistung", 1.0),
        ("hp_power", "Wärmepumpe", "Leistung", 1.0),
    ]
    n = max((len(timeline.get(f) or []) for f, _, _, _ in series), default=0)
    hours = hours_axis(n)

    rows: list[dict[str, Any]] = []
    for field, label, group, factor in series:
        for i, value in enumerate(timeline.get(field) or []):
            rows.append({
                "Stunde": float(hours[i]),
                "Uhrzeit": format_hour(float(hours[i])),
                "reihe": field,
                "Reihe": label,
                "Gruppe": group,
                "Wert": float(value) * factor,
            })
    return pd.DataFrame(rows, columns=["Stunde", "Uhrzeit", "reihe", "Reihe", "Wert", "Gruppe"])


def curtailment_steps(timeline: dict[str, Any]) -> pd.DataFrame:
    """The quarter hours in which §14a curtailment was active, for shading."""
    flags = timeline.get("curtailment") or []
    hours = hours_axis(len(flags))
    rows = [{"Stunde": float(hours[i]), "Uhrzeit": format_hour(float(hours[i]))}
            for i, on in enumerate(flags) if on]
    return pd.DataFrame(rows, columns=["Stunde", "Uhrzeit"])


# ══════════════════════════════════════════════════════════════
# Overload map: WHERE the grid suffered, not only how often
# ══════════════════════════════════════════════════════════════
#: Loading (p.u.) → colour. A real LV grid is cable-limited long before the
#: transformer notices, so the map has to make a 0.9 p.u. cable visible.
_LOAD_COLORS = ((1.0, "#d7191c", 5.0), (0.9, "#fdae61", 4.0), (0.7, "#ffd54f", 3.0), (0.0, "#7cb342", 2.5))


def _load_style(pu: float, tripped: bool = False) -> tuple[str, float]:
    """(colour, stroke width) for a loading in p.u.

    `tripped` erzwingt die Überlastungsfarbe. Der gespeicherte Spitzenwert ist ein
    MITTELWERT über die Seeds: ein Kabel, das in zwei von sechs Läufen seine Grenze
    verletzt hat, liegt im Mittel unter 1.0. Es muss trotzdem als Verletzung erkennbar
    sein, sonst würde die Karte der danebenstehenden Überlastungszahl widersprechen.
    """
    if tripped:
        return _LOAD_COLORS[0][1], _LOAD_COLORS[0][2]
    for floor, color, weight in _LOAD_COLORS:
        if pu > floor:
            return color, weight
    return "#7cb342", 2.5


def render_overload_map(network, summary, key: str = "overload_map") -> None:
    """Draw the grid, highlighting the cables that ran hot or tripped.

    `network` ist das gespeicherte GridNetwork des Laufs; `summary` liefert die
    pro Leitung gespeicherte Spitzenauslastung je Szenario. Nicht aufgeführte
    Leitungen blieben unter der Warnschwelle und werden blass im Hintergrund
    gezeichnet, damit der Blick direkt zum kritischen Pfad geht.
    """
    import folium
    import streamlit as st
    from streamlit_folium import st_folium

    if network is None or not getattr(network, "buses", None):
        st.info("Kein gespeichertes Netz für diesen Lauf. Die Überlastungskarte benötigt `network.json`.")
        return

    rows = [r for r in summary if r.get("line_peak_loading_pu")]
    if not rows:
        st.info("Dieser Lauf stammt aus der Zeit vor der Leitungs-Aufzeichnung, daher gibt es "
                "keine Überlastungskarte dafür. Neue Läufe speichern sie automatisch.")
        return

    labels = [r["scenario"] for r in rows]
    chosen = st.selectbox("Szenario", labels, index=len(labels) - 1, key=f"{key}_scenario", format_func=theme.scenario_label)
    row = next(r for r in rows if r["scenario"] == chosen)
    peaks: dict = row.get("line_peak_loading_pu") or {}
    steps: dict = row.get("line_overload_steps") or {}

    coords = {b.bus_id: (b.y_coord, b.x_coord)
              for b in network.buses if b.x_coord is not None and b.y_coord is not None}
    if not coords:
        st.info("Das gespeicherte Netz hat keine Koordinaten und kann daher nicht auf einer Karte dargestellt werden.")
        return

    centre = (float(np.mean([c[0] for c in coords.values()])),
              float(np.mean([c[1] for c in coords.values()])))
    
    import xyzservices.providers as xyz_providers
    m = folium.Map(location=centre, zoom_start=15, tiles=xyz_providers.Esri.WorldGrayCanvas)

    n_over = 0
    for ln in network.lines:
        a, b = coords.get(ln.from_bus), coords.get(ln.to_bus)
        if not a or not b:
            continue
        pu = peaks.get(ln.line_id)
        if pu is None:

            folium.PolyLine(
                [a, b], color="#8a8f98", weight=1.6, opacity=0.8,
                tooltip=f"{ln.line_id}: unauffällig (unter der Beobachtungsschwelle)",
            ).add_to(m)
            continue
        trips = steps.get(ln.line_id, 0)
        color, weight = _load_style(pu, tripped=trips > 0)
        n_over += trips > 0
        folium.PolyLine(
            [a, b], color=color, weight=weight, opacity=0.95,
            tooltip=f"{ln.line_id}: Spitzenwert {pu:.2f} p.u." + (f", {trips:.1f} Schritte überlastet" if trips else ""),
        ).add_to(m)

    for t in network.transformers:
        tb = coords.get(t.lv_bus)
        if tb:
            folium.Marker(tb, icon=folium.Icon(color="black", icon="bolt", prefix="fa"),
                          tooltip=f"{t.trafo_id}: {t.s_nom_mva*1000:.0f} kVA").add_to(m)

    worst = max(peaks.values()) if peaks else 0.0
    st.caption(
        f"**{theme.scenario_short(chosen)}:** schlechtestes Kabel {worst:.2f} p.u. · {n_over} Kabel über der Nennlast. "
        "🟥 >1,0 überlastet · 🟧 >0,9 · 🟨 >0,7 · 🟩 belastet, aber unauffällig · grau = unter Warnschwelle. "
        "⚡ = Transformator. Zum Anzeigen des Spitzenwerts über ein Segment fahren."
    )
    st_folium(m, height=520, width=None, returned_objects=[], key=key)


# ══════════════════════════════════════════════════════════════
# Streamlit view
# ══════════════════════════════════════════════════════════════
_HOUR_AXIS = dict(
    values=[12, 15, 18, 21, 24, 27, 30, 33, 36],
    labelExpr="format(datum.value % 24, '02') + ':00'",
)


def _scenario_scale(frame: pd.DataFrame):  # pragma: no cover (UI)
    import altair as alt

    present = [s for s in theme.SCENARIO_ORDER if s in set(frame["scenario"])]
    present += [s for s in dict.fromkeys(frame["scenario"]) if s not in present]
    return alt.Scale(domain=[theme.scenario_label(s) for s in present],
                     range=[theme.SCENARIO_COLORS.get(s, "#6c757d") for s in present])


def _metric_chart(frame: pd.DataFrame, kennzahl: str):  # pragma: no cover (UI)
    """One metric, the scenarios as bars in the scenario palette.

    One chart per metric rather than one faceted chart, because `width:
    "container"` only applies to single-view and layered specs; inside a facet
    or concat the child views cannot size themselves and render empty until the
    viewer opens them fullscreen.
    """
    import altair as alt

    data = frame[frame["Kennzahl"] == kennzahl]
    scale = _scenario_scale(frame)
    order = list(scale.domain)
    base = alt.Chart(data).encode(
        y=alt.Y("Szenario:N", title=None, sort=order),
    )
    bars = base.mark_bar(height=20, cornerRadiusEnd=3).encode(
        x=alt.X("Wert:Q", title=data["Einheit"].iloc[0] if not data.empty else None),
        color=alt.Color("Szenario:N", scale=scale, legend=None),
        tooltip=[alt.Tooltip("Szenario:N"), alt.Tooltip("Kennzahl:N"),
                 alt.Tooltip("Wert:Q", format=".2f"),
                 alt.Tooltip("Streuung:Q", format=".2f", title="± Streuung"),
                 alt.Tooltip("Einheit:N")],
    )
    # the spread over the seeds: a gap narrower than these whiskers is noise
    spread = base.mark_rule(strokeWidth=1.6, color=theme.SERIES_COLORS["transformer"],
                            opacity=0.8).encode(
        x=alt.X("untere:Q", title=None), x2="obere:Q",
    ).transform_calculate(
        untere="max(0, datum.Wert - datum.Streuung)",
        obere="datum.Wert + datum.Streuung",
    )
    labels = base.mark_text(align="left", dx=6, fontSize=11,
                            color=theme.SERIES_COLORS["transformer"]).encode(
        x=alt.X("obere:Q", title=None), text=alt.Text("Wert:Q", format=".2f"),
    ).transform_calculate(obere="datum.Wert + datum.Streuung")
    return (bars + spread + labels).properties(width="container", height=alt.Step(30))


#: (group, y-axis title, {series: colour}, height, limit line); one chart each.
EPISODE_PANELS: tuple[tuple[str, str, dict[str, str], int, float | None], ...] = (
    ("Auslastung", "Auslastung (%)",
     {"Trafo-Auslastung": theme.SERIES_COLORS["transformer"],
      "max. Leitungsauslastung": theme.SERIES_COLORS["line"]}, 220, 100.0),
    ("Leistung", "Leistung (kW)",
     {"EV": theme.DEVICE_COLORS["ev"], "Batterie": theme.DEVICE_COLORS["battery"],
      "Wärmepumpe": theme.DEVICE_COLORS["hp"], "PV-Erzeugung": theme.DEVICE_COLORS["pv"]}, 220, None),
    ("Preis", "Preis (€/kWh)", {"Strompreis": theme.SERIES_COLORS["line"]}, 150, None),
)


def _episode_panel(frame: pd.DataFrame, curtailed: pd.DataFrame,
                   group: str, y_title: str, colors: dict[str, str],
                   height: int, rule: float | None):  # pragma: no cover (UI)
    """One panel of the representative day.

    Separate charts rather than one vconcat: `width: "container"` applies only
    to single-view and layered specs, so a concatenated child cannot size to the
    page and renders empty at normal width.
    """
    import altair as alt

    data = frame[frame["Gruppe"] == group]
    present = [r for r in colors if r in set(data["Reihe"])]
    layers = []
    if not curtailed.empty:
        # every quarter hour the grid operator had to intervene
        layers.append(
            alt.Chart(curtailed).mark_rule(color=theme.STATUS_COLORS["kritisch"],
                                           opacity=0.13, strokeWidth=4)
            .encode(x=alt.X("Stunde:Q"))
        )
    if rule is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"y": [rule]}))
            .mark_rule(color=theme.STATUS_COLORS["kritisch"], strokeDash=[5, 4], strokeWidth=1.3)
            .encode(y="y:Q")
        )
    layers.append(
        alt.Chart(data).mark_line(strokeWidth=2).encode(
            x=alt.X("Stunde:Q", title="Uhrzeit", axis=alt.Axis(**_HOUR_AXIS)),
            y=alt.Y("Wert:Q", title=y_title),
            color=alt.Color("Reihe:N",
                            scale=alt.Scale(domain=present, range=[colors[r] for r in present]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("Uhrzeit:N"), alt.Tooltip("Reihe:N"),
                     alt.Tooltip("Wert:Q", format=".2f")],
        )
    )
    return alt.layer(*layers).properties(width="container", height=height)


def render_comparison(summary: list[dict[str, Any]], timelines: list[dict[str, Any]] | None = None,
                     network=None, key: str = "vergleich") -> None:  # pragma: no cover (UI)
    """The comparison tab: all four strategies on the same grid."""
    import streamlit as st

    if not summary:
        st.info("Keine Ergebnisse vorhanden. Zuerst ein Experiment ausführen.")
        return

    # The two selectors are split across the tab, each next to what it changes.
    # The Ausstattungsgrad decides which run EVERYTHING here describes, so it
    # stays at the top; the scenario only picks the episode at the bottom, and
    # sitting up here it read as though it changed the metric charts too.
    # Both still read and write the state shared with every other tab.
    chosen_pen = penetration_picker(timelines or [], key)
    if chosen_pen is None:
        chosen_pen = max((float(r.get("penetration", 0.0)) for r in summary), default=0.0)

    # ── Kennzahlen nebeneinander ──────────────────────────────
    metrics = metric_frame(summary, chosen_pen)
    if metrics.empty:
        st.info("Die Ergebnisse enthalten keine Vergleichskennzahlen.")
    else:
        st.subheader("Kennzahlen je Szenario")
        st.caption(
            "Dieselbe Netzsituation, vier Regelstrategien. Der Strich in jedem Balken "
            "ist die Streuung über die Auswertungsläufe: Unterschiede, die kleiner "
            "sind als dieser Strich, sind keine. **§14a-Eingriffe** und **Stromkosten** "
            "sind besser, wenn sie klein sind; **EV-Ziel** und **WP-Komfort**, wenn sie "
            "groß sind: eine Strategie, die das Netz schont und das Auto leer lässt, "
            "hat nichts gewonnen."
        )
        for _, label, _, higher_better in METRICS:
            if label not in set(metrics["Kennzahl"]):
                continue
            st.markdown(f"**{label}**: {'größer ist besser' if higher_better else 'kleiner ist besser'}")
            st.altair_chart(_metric_chart(metrics, label), width="stretch")
        download_pair(metrics.drop(columns=["scenario", "kennzahl"]),
                      "Kennzahlen je Szenario", f"{key}_metrics")

    # ── One representative episode ───────────────────────────
    if timelines:
        scenario = scenario_picker(timelines, key)
        st.subheader(f"Eine repräsentative 24-Stunden-Episode: {theme.scenario_short(scenario)}"
                     if scenario else "Eine repräsentative 24-Stunden-Episode")
        timeline = select_timeline(timelines, scenario, chosen_pen) if scenario else None
        if timeline is None:
            st.info("Für diese Kombination liegt keine Episode vor.")
        else:
            st.caption(
                "Ein einzelner Tag, von Mittag bis Mittag, für das gewählte Szenario. "
                "Rot hinterlegt sind die Viertelstunden, in denen §14a gegriffen hat. "
                "Einzelne Haushalte stehen im Reiter **Geräte & Haushalte**."
            )
            frame = episode_frame(timeline)
            curtailed = curtailment_steps(timeline)
            for group, y_title, colors, height, rule in EPISODE_PANELS:
                if group not in set(frame["Gruppe"]):
                    continue
                st.altair_chart(
                    _episode_panel(frame, curtailed, group, y_title, colors, height, rule),
                    width="stretch",
                )
            download_pair(frame, "Episode", f"{key}_episode")

    # ── Überlastungskarte ─────────────────────────────────────
    if network is not None:
        st.subheader("Wo das Netz überlastet war")
        render_overload_map(network, summary)
