# dashboard/auslastung.py
# ─────────────────────────────────────────────────────────────
# Netzauslastung in Viertelstundenwerten — der Auslastungs-Reiter.
#
# Reads only fields run_experiment.py already writes to timelines.json:
#   transformer_loading, max_line_loading, curtailment,
#   overloaded_transformers, overloaded_lines
#
# Why BOTH loading series are always drawn together: a low-voltage grid is
# cable-limited long before the transformer notices. In the 17-household
# reference run the transformer sits at a comfortable-looking 80 % while the
# service cables run at 295 % — showing the transformer figure alone reads
# "unauffällig" straight through a threefold thermal violation.
#
# Everything above the first section is Streamlit-free so it can be unit-tested.
#
# UI text is German; the timeline keys stay English — they are the raw field
# names from timelines.json and translating them would break every lookup.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from dashboard import theme
from dashboard.controls import scenario_penetration_picker
from dashboard.export import download_pair

EPISODE_START_HOUR = 12
TIMESTEP_HOURS = 0.25


# ══════════════════════════════════════════════════════════════
# Reine Helfer (kein Streamlit — unit-testbar)
# ══════════════════════════════════════════════════════════════
def format_hour(hour: float) -> str:
    """14.25 → "14:15". Modulo 24 because the episode runs noon → noon."""
    hh = int(hour) % 24
    mm = int(round((hour - int(hour)) * 60)) % 60
    return f"{hh:02d}:{mm:02d}"


def hours_axis(n: int) -> np.ndarray:
    """Episode-local hour for each of the n quarter-hour steps (starts at noon)."""
    return EPISODE_START_HOUR + np.arange(n) * TIMESTEP_HOURS


def utilization_frame(timeline: dict[str, Any]) -> pd.DataFrame:
    """Long-form frame for the daily-profile chart.

    One row per (step, series) with the loading in PERCENT — percent rather than
    p.u. because the whole point of the view is "x von 100 %", and a chart that
    silently mixes 0.78 with 78 invites exactly the misreading we are avoiding.
    """
    trafo = [float(v) * 100.0 for v in timeline.get("transformer_loading", [])]
    line = [float(v) * 100.0 for v in timeline.get("max_line_loading", [])]
    n = max(len(trafo), len(line))
    hours = hours_axis(n)

    rows: list[dict[str, Any]] = []
    for key, values in (("transformer", trafo), ("line", line)):
        for i, value in enumerate(values):
            rows.append({
                "Schritt": i,
                "Stunde": float(hours[i]),
                "Uhrzeit": format_hour(float(hours[i])),
                "Messgröße": theme.SERIES_LABELS_DE[key],
                "reihe": key,
                "Auslastung": value,
            })
    return pd.DataFrame(rows)


def duration_curve(values: Sequence[float]) -> pd.DataFrame:
    """Auslastungsdauerlinie: values sorted descending against cumulative hours.

    Answers "how many hours above X %" by reading straight off the curve —
    the standard grid-planning view of the same data as the daily profile.
    """
    percent = np.sort(np.asarray([float(v) * 100.0 for v in values]))[::-1]
    return pd.DataFrame({
        "Stunden": (np.arange(len(percent)) + 1) * TIMESTEP_HOURS,
        "Auslastung": percent,
    })


def hours_above(values: Iterable[float], threshold_pu: float = theme.OVERLOAD_PU) -> float:
    """Hours (not steps) strictly above a threshold in p.u.

    Strictly above, to match how the backend flags an element as overloaded.
    """
    return sum(1 for v in values if float(v) > threshold_pu) * TIMESTEP_HOURS


def peak_moment(values: Sequence[float]) -> tuple[float, str]:
    """(peak in percent, clock time it occurred) — empty input gives (0.0, "—")."""
    if len(values) == 0:
        return 0.0, theme.NO_VALUE
    arr = np.asarray([float(v) for v in values])
    idx = int(np.argmax(arr))
    return float(arr[idx]) * 100.0, format_hour(float(hours_axis(len(arr))[idx]))


def overload_matrix(timeline: dict[str, Any]) -> pd.DataFrame:
    """Element × step overload flags from the exported ID lists.

    `overloaded_lines`/`overloaded_transformers` are lists of the IDs over the
    limit at each step, so the result is binary: which element, when, how long.
    A graded version would need the per-element loadings, which are not exported.
    Elements are ordered by how long they were overloaded, worst first.
    """
    per_step: list[tuple[str, list[str]]] = []
    for field, kind in (("overloaded_lines", "Leitung"), ("overloaded_transformers", "Trafo")):
        steps = timeline.get(field) or []
        per_step.append((kind, steps))

    n = max((len(steps) for _, steps in per_step), default=0)
    flags: dict[tuple[str, str], np.ndarray] = {}
    for kind, steps in per_step:
        for i, ids in enumerate(steps):
            for element_id in ids or []:
                key = (kind, str(element_id))
                if key not in flags:
                    flags[key] = np.zeros(n, dtype=bool)
                flags[key][i] = True

    if not flags:
        return pd.DataFrame(columns=["Element", "Typ", "Schritt", "Stunde", "Uhrzeit", "überlastet"])

    order = sorted(flags, key=lambda k: (-int(flags[k].sum()), k[1]))
    hours = hours_axis(n)
    rows: list[dict[str, Any]] = []
    for kind, element_id in order:
        mask = flags[(kind, element_id)]
        for i in range(n):
            rows.append({
                "Element": element_id,
                "Typ": kind,
                "Schritt": i,
                "Stunde": float(hours[i]),
                "Uhrzeit": format_hour(float(hours[i])),
                "überlastet": bool(mask[i]),
            })
    return pd.DataFrame(rows)


_TABLE_COLUMNS = ["Element", "Typ", "Dauer gesamt (h)", "Abschnitte",
                  "längster Abschnitt", "Dauer davon (h)", "Zeitfenster"]


def _format_span(start_hour: float, end_hour: float) -> str:
    """'13:00–13:15' for a span, or just '13:00' when it lasted one step."""
    start, end = format_hour(float(start_hour)), format_hour(float(end_hour))
    return start if start == end else f"{start}–{end}"


def contiguous_blocks(steps: list[int]) -> list[tuple[int, int]]:
    """Consecutive runs in a sorted list of step indices, as (first, last) pairs.

    An element rarely stays over its limit in one stretch: it trips when the
    load peaks, recovers, and trips again. Those separate episodes are what the
    table has to show.
    """
    if not steps:
        return []
    blocks: list[tuple[int, int]] = []
    start = previous = steps[0]
    for step in steps[1:]:
        if step != previous + 1:
            blocks.append((start, previous))
            start = step
        previous = step
    blocks.append((start, previous))
    return blocks


def overload_table(timeline: dict[str, Any]) -> pd.DataFrame:
    """Per overloaded element: how long in total, in how many separate stretches.

    "Dauer gesamt" is the SUM of the overloaded quarter hours, which is not the
    distance between the first and the last one: an element that trips at 13:00
    and again at 08:45 spans twenty hours while being overloaded for six of
    them. The earlier table showed only that span as "von … bis …", which reads
    as one continuous period and overstated every entry. It now reports the sum,
    how many separate stretches it took, and the longest single one — the figure
    that decides whether a cable had time to cool down.
    """
    matrix = overload_matrix(timeline)
    if matrix.empty:
        return pd.DataFrame(columns=_TABLE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for (element_id, kind), group in matrix.groupby(["Element", "Typ"], sort=False):
        hit = group[group["überlastet"]]
        if hit.empty:
            continue
        steps = sorted(int(s) for s in hit["Schritt"])
        hours = dict(zip(group["Schritt"], group["Stunde"]))
        blocks = contiguous_blocks(steps)
        longest = max(blocks, key=lambda b: b[1] - b[0])
        rows.append({
            "Element": element_id,
            "Typ": kind,
            "Dauer gesamt (h)": round(len(steps) * TIMESTEP_HOURS, 2),
            "Abschnitte": len(blocks),
            "längster Abschnitt": _format_span(hours[longest[0]], hours[longest[1]]),
            "Dauer davon (h)": round((longest[1] - longest[0] + 1) * TIMESTEP_HOURS, 2),
            "Zeitfenster": _format_span(hours[steps[0]], hours[steps[-1]]),
        })
    return (pd.DataFrame(rows, columns=_TABLE_COLUMNS)
            .sort_values("Dauer gesamt (h)", ascending=False, ignore_index=True))


def headline(timeline: dict[str, Any]) -> dict[str, Any]:
    """The numbers the verdict line and KPI row are built from."""
    trafo = timeline.get("transformer_loading") or []
    line = timeline.get("max_line_loading") or []
    trafo_peak, trafo_at = peak_moment(trafo)
    line_peak, line_at = peak_moment(line)
    table = overload_table(timeline)
    return {
        "trafo_peak_percent": trafo_peak,
        "trafo_peak_at": trafo_at,
        "trafo_mean_percent": float(np.mean(trafo) * 100.0) if len(trafo) else 0.0,
        "line_peak_percent": line_peak,
        "line_peak_at": line_at,
        "hours_over": hours_above(line) + hours_above(trafo),
        "curtailment_steps": int(sum(1 for c in (timeline.get("curtailment") or []) if c)),
        "n_elements_overloaded": int(len(table)),
        "worst_element": (table.iloc[0]["Element"] if not table.empty else None),
        "status": theme.status_of(max(
            max((float(v) for v in trafo), default=0.0),
            max((float(v) for v in line), default=0.0),
        )),
    }


def select_timeline(timelines: list[dict[str, Any]], scenario: str, penetration: float) -> dict[str, Any] | None:
    """The one timeline record for a scenario × penetration, or None."""
    for record in timelines:
        if record.get("scenario") == scenario and abs(float(record.get("penetration", -1)) - penetration) < 1e-9:
            return record
    return None


# ══════════════════════════════════════════════════════════════
# Streamlit-Ansicht
# ══════════════════════════════════════════════════════════════
def _band_frame(scale_max: float) -> pd.DataFrame:
    """Severity bands clipped to the chart's y-range, as chart data."""
    rows = []
    for lower, upper, key, label, color in theme.LOAD_BANDS:
        low = lower * 100.0
        high = scale_max if upper == float("inf") else upper * 100.0
        if low >= scale_max:
            continue
        rows.append({"von": low, "bis": min(high, scale_max),
                     "Bewertung": label, "farbe": color, "schluessel": key})
    return pd.DataFrame(rows)


def _scale_max(timeline: dict[str, Any]) -> float:
    """Y-axis top: enough headroom for the data, never below the 120 % baseline."""
    values = list(timeline.get("transformer_loading") or []) + list(timeline.get("max_line_loading") or [])
    peak = max((float(v) * 100.0 for v in values), default=0.0)
    return max(theme.LOAD_SCALE_MIN_PERCENT, peak * 1.08)


def _daily_profile_chart(timeline: dict[str, Any]):  # pragma: no cover (UI)
    import altair as alt

    frame = utilization_frame(timeline)
    scale_max = _scale_max(timeline)
    y_scale = alt.Scale(domain=[0, scale_max], nice=False)

    bands = (
        alt.Chart(_band_frame(scale_max))
        .mark_rect(opacity=0.16)
        .encode(
            y=alt.Y("von:Q", scale=y_scale, title="Auslastung (%)"),
            y2="bis:Q",
            color=alt.Color("farbe:N", scale=None, legend=None),
        )
    )
    limit = (
        alt.Chart(pd.DataFrame({"y": [100.0]}))
        .mark_rule(color=theme.STATUS_COLORS["kritisch"], strokeDash=[5, 4], strokeWidth=1.4)
        .encode(y=alt.Y("y:Q", scale=y_scale))
    )
    lines = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("Stunde:Q", title="Uhrzeit",
                    axis=alt.Axis(values=[12, 15, 18, 21, 24, 27, 30, 33, 36],
                                  labelExpr="format(datum.value % 24, '02') + ':00'")),
            y=alt.Y("Auslastung:Q", scale=y_scale, title="Auslastung (%)"),
            color=alt.Color(
                "Messgröße:N",
                scale=alt.Scale(
                    domain=[theme.SERIES_LABELS_DE["transformer"], theme.SERIES_LABELS_DE["line"]],
                    range=[theme.SERIES_COLORS["transformer"], theme.SERIES_COLORS["line"]],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[alt.Tooltip("Uhrzeit:N"),
                     alt.Tooltip("Messgröße:N"),
                     alt.Tooltip("Auslastung:Q", format=".1f", title="Auslastung (%)")],
        )
    )
    return (bands + limit + lines).properties(width="container", height=340)


def _duration_chart(timeline: dict[str, Any]):  # pragma: no cover (UI)
    import altair as alt

    frame = duration_curve(timeline.get("max_line_loading") or [])
    scale_max = _scale_max(timeline)
    y_scale = alt.Scale(domain=[0, scale_max], nice=False)
    area = (
        alt.Chart(frame)
        .mark_area(opacity=0.22, line={"strokeWidth": 2}, color=theme.SERIES_COLORS["line"])
        .encode(
            x=alt.X("Stunden:Q", title="Stunden über diesem Wert",
                    scale=alt.Scale(domain=[0, 24], nice=False)),
            y=alt.Y("Auslastung:Q", scale=y_scale, title="Auslastung (%)"),
            tooltip=[alt.Tooltip("Stunden:Q", format=".2f"),
                     alt.Tooltip("Auslastung:Q", format=".1f", title="Auslastung (%)")],
        )
    )
    limit = (
        alt.Chart(pd.DataFrame({"y": [100.0]}))
        .mark_rule(color=theme.STATUS_COLORS["kritisch"], strokeDash=[5, 4], strokeWidth=1.4)
        .encode(y=alt.Y("y:Q", scale=y_scale))
    )
    return (area + limit).properties(width="container", height=260)


def _overload_chart(timeline: dict[str, Any]):  # pragma: no cover (UI)
    import altair as alt

    frame = overload_matrix(timeline)
    frame = frame[frame["überlastet"]]
    order = list(dict.fromkeys(frame["Element"]))
    return (
        alt.Chart(frame)
        .mark_rect(height=13)
        .encode(
            x=alt.X("Stunde:Q", title="Uhrzeit",
                    scale=alt.Scale(domain=[12, 36], nice=False),
                    axis=alt.Axis(values=[12, 15, 18, 21, 24, 27, 30, 33, 36],
                                  labelExpr="format(datum.value % 24, '02') + ':00'")),
            y=alt.Y("Element:N", title=None, sort=order),
            color=alt.value(theme.STATUS_COLORS["kritisch"]),
            tooltip=[alt.Tooltip("Element:N"), alt.Tooltip("Typ:N"), alt.Tooltip("Uhrzeit:N")],
        )
        # Per-row step rather than a pixel total, for the same reason as the
        # overview matrix: a pixel height loses against the container fit and
        # the rows collapse into each other.
        .properties(width="container", height=alt.Step(22))
    )


def render_auslastung(timelines: list[dict[str, Any]], key: str = "auslastung") -> None:  # pragma: no cover (UI)
    """Der Auslastungs-Reiter: Tagesgang, Überlast-Matrix, Dauerlinie."""
    import streamlit as st

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

    head = headline(timeline)

    # ── Urteil ────────────────────────────────────────────────
    worst = head["worst_element"]
    hours_over = f"{head['hours_over']:.2f}".replace(".", ",")
    if head["status"] == "kritisch":
        st.error(
            f"**Überlast:** Spitze {head['line_peak_percent']:.0f} % um {head['line_peak_at']} Uhr"
            + (f", schlimmstes Element `{worst}`" if worst else "")
            + f". Insgesamt {hours_over} h über 100 %."
        )
    elif head["status"] == "grenzbereich":
        st.warning(f"**Grenzbereich:** Spitze {head['line_peak_percent']:.0f} % um "
                   f"{head['line_peak_at']} Uhr, aber keine Überschreitung.")
    else:
        st.success(f"**Unkritisch:** Spitze {head['line_peak_percent']:.0f} % um "
                   f"{head['line_peak_at']} Uhr.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spitze Leitung", f"{head['line_peak_percent']:.0f} %", help=f"um {head['line_peak_at']} Uhr")
    c2.metric("Spitze Trafo", f"{head['trafo_peak_percent']:.0f} %",
              help=f"um {head['trafo_peak_at']} Uhr · Ø {head['trafo_mean_percent']:.0f} % über 24 h")
    c3.metric("Stunden über 100 %", f"{head['hours_over']:.2f} h".replace(".", ","))
    c4.metric("§14a-Eingriffe", head["curtailment_steps"], help="Viertelstunden mit Abregelung")

    # ── Tagesgang ─────────────────────────────────────────────
    st.subheader("Tagesverlauf")
    st.caption(
        "Beide Reihen gehören zusammen: im Niederspannungsnetz bindet fast immer das "
        "**Kabel** zuerst, nicht der Transformator. Ein entspannt wirkender Trafo-Wert "
        "kann eine deutliche thermische Verletzung im Strang verdecken."
    )
    st.altair_chart(_daily_profile_chart(timeline), width="stretch")
    download_pair(utilization_frame(timeline), "Tagesverlauf", f"{key}_profil")

    gap = head["line_peak_percent"] - head["trafo_peak_percent"]
    if gap > 20:
        st.caption(
            f"↳ Hier liegen **{gap:.0f} Prozentpunkte** zwischen Kabel- und Trafo-Spitze: "
            "die Last verteilt sich sehr ungleich über die Stränge."
        )

    # ── Überlast-Matrix ───────────────────────────────────────
    st.subheader("Welche Elemente wann überlastet waren")
    table = overload_table(timeline)
    if table.empty:
        st.success("Kein Element war in diesem Lauf über seiner Nennleistung.")
    else:
        st.caption(
            "Binär: überlastet oder nicht. Abgestufte Farben bräuchten die Auslastung "
            "je Element, die der Export derzeit nicht enthält. **Ein Element ist selten "
            "am Stück überlastet** — es geht über die Grenze, erholt sich und trippt "
            "erneut. Die Tabelle nennt deshalb die aufsummierte Dauer, in wie vielen "
            "getrennten Abschnitten sie zustande kam, und den längsten einzelnen davon; "
            "**Zeitfenster** ist nur die Spanne vom ersten bis zum letzten Auftreten, "
            "nicht die Zeit dazwischen."
        )
        st.altair_chart(_overload_chart(timeline), width="stretch")
        st.dataframe(table, width="stretch", hide_index=True)
        download_pair(table, "Überlast-Matrix", f"{key}_ueberlast")

    # ── Dauerlinie ────────────────────────────────────────────
    st.subheader("Auslastungsdauerlinie")
    st.caption(
        "Dieselben Werte, absteigend sortiert: direkt ablesbar, wie viele Stunden "
        "über einer Grenze lagen. Bezieht sich auf die stärkstbelastete Leitung."
    )
    st.altair_chart(_duration_chart(timeline), width="stretch")
    download_pair(duration_curve(timeline.get("max_line_loading") or []),
                  "Dauerlinie", f"{key}_dauerlinie")
