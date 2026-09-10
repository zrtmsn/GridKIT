# dashboard/ueberblick.py
# ─────────────────────────────────────────────────────────────
# Überblick — die Antwort, bevor jemand scrollt.
#
# Answers the project's own question — does §14a curtailment hold up with EV,
# battery, heat pump and PV on the same low-voltage grid? — across ALL
# scenarios and penetration levels at once. The other tabs then explain one
# combination at a time.
#
# Reads summary.json for the seed-averaged figures and falls back to
# timelines.json where a run predates a field: the cable peak (line_peak_max)
# only exists in summaries written after the two producers were unified, and an
# overview that crashes on an older run is worse than one that quietly uses the
# representative episode instead.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any

import pandas as pd

from dashboard import theme
from dashboard.auslastung import select_timeline
from dashboard.export import download_pair

#: Order of severity, worst first — used to reduce many combinations to one verdict.
_SEVERITY = ["kritisch", "grenzbereich", "warnung", "gut"]


# ══════════════════════════════════════════════════════════════
# Reine Helfer (kein Streamlit — unit-testbar)
# ══════════════════════════════════════════════════════════════
def cable_peak_pu(record: dict[str, Any], timelines: list[dict[str, Any]] | None = None) -> float | None:
    """Worst cable loading for one summary row, in p.u.

    Prefers `line_peak_max` (averaged over all evaluation seeds). Older runs do
    not carry it, so it falls back to the representative episode in
    timelines.json — a weaker figure, but a real one. None when neither exists.
    """
    value = record.get("line_peak_max")
    if value is not None:
        return float(value)
    for timeline in timelines or []:
        if (timeline.get("scenario") == record.get("scenario")
                and abs(float(timeline.get("penetration", -1)) - float(record.get("penetration", -2))) < 1e-9):
            loadings = timeline.get("max_line_loading") or []
            return max((float(v) for v in loadings), default=None) if loadings else None
    return None


def overview_frame(summary: list[dict[str, Any]],
                   timelines: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """One row per scenario × penetration with the peaks and the resulting status.

    The status is taken from whichever of transformer and cable is worse,
    because either one over its limit is an overload — reporting only the
    transformer is exactly the misreading this dashboard exists to prevent.
    """
    rows: list[dict[str, Any]] = []
    for record in summary or []:
        scenario = record.get("scenario", "")
        penetration = float(record.get("penetration", 0.0))
        trafo = record.get("peak_mean")
        trafo_pu = float(trafo) if trafo is not None else None
        cable_pu = cable_peak_pu(record, timelines)
        worst = max([v for v in (trafo_pu, cable_pu) if v is not None], default=None)
        rows.append({
            "scenario": scenario,
            "Szenario": theme.scenario_label(scenario),
            "penetration": penetration,
            "EV-Anteil": f"{penetration:.0%}",
            "Trafo": trafo_pu,
            "Kabel": cable_pu,
            "Spitze": worst,
            "status": theme.status_of(worst) if worst is not None else None,
            "curtailment": record.get("curtailment_mean"),
            "soc": record.get("soc_mean"),
        })
    return pd.DataFrame(rows, columns=["scenario", "Szenario", "penetration", "EV-Anteil",
                                       "Trafo", "Kabel", "Spitze", "status",
                                       "curtailment", "soc"])


def worst_status(frame: pd.DataFrame) -> str | None:
    """The most severe status anywhere in the frame."""
    present = set(frame["status"].dropna()) if not frame.empty else set()
    for level in _SEVERITY:
        if level in present:
            return level
    return None


def breaking_point(frame: pd.DataFrame) -> float | None:
    """Lowest EV share at which any scenario goes over the limit, or None.

    The single most useful number in the whole dashboard for planning: not
    "does it break" but "how much adoption does this grid take before it does".
    """
    over = frame[frame["status"] == "kritisch"] if not frame.empty else frame
    return float(over["penetration"].min()) if not over.empty else None


def safe_ceiling(frame: pd.DataFrame) -> float | None:
    """Highest EV share at which NO scenario goes over the limit."""
    if frame.empty:
        return None
    per_pen = frame.groupby("penetration")["status"].apply(
        lambda s: "kritisch" in set(s.dropna())
    )
    safe = [float(p) for p, breaks in per_pen.items() if not breaks]
    return max(safe) if safe else None


def scenario_ranking(frame: pd.DataFrame, penetration: float) -> pd.DataFrame:
    """Scenarios at one penetration, gentlest peak first."""
    if frame.empty:
        return frame
    at = frame[(frame["penetration"] - penetration).abs() < 1e-9].copy()
    return at.sort_values("Spitze", na_position="last", ignore_index=True)


def headline_numbers(frame: pd.DataFrame) -> dict[str, Any]:
    """The figures the verdict banner and KPI row read."""
    if frame.empty or frame["Spitze"].dropna().empty:
        return {"worst_peak": None, "worst_scenario": None, "worst_penetration": None,
                "status": None, "breaking_point": None, "safe_ceiling": None,
                "n_over": 0, "n_total": int(len(frame))}
    worst_row = frame.loc[frame["Spitze"].idxmax()]
    return {
        "worst_peak": float(worst_row["Spitze"]),
        "worst_scenario": worst_row["Szenario"],
        "worst_penetration": float(worst_row["penetration"]),
        "status": worst_status(frame),
        "breaking_point": breaking_point(frame),
        "safe_ceiling": safe_ceiling(frame),
        "n_over": int((frame["status"] == "kritisch").sum()),
        "n_total": int(len(frame)),
    }


# ══════════════════════════════════════════════════════════════
# Streamlit-Ansicht
# ══════════════════════════════════════════════════════════════
def _matrix_chart(frame: pd.DataFrame):  # pragma: no cover (UI)
    import altair as alt

    data = frame.dropna(subset=["Spitze"]).copy()
    data["Prozent"] = data["Spitze"] * 100.0
    data["Bewertung"] = data["status"].map(theme.STATUS_LABELS_DE)
    scenarios = [s for s in theme.SCENARIO_ORDER if s in set(data["scenario"])]
    order = [theme.scenario_label(s) for s in scenarios] or list(data["Szenario"])

    base = alt.Chart(data).encode(
        x=alt.X("EV-Anteil:N", title="EV-Anteil", sort=sorted(set(data["EV-Anteil"]))),
        y=alt.Y("Szenario:N", title=None, sort=order),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=[k for k in _SEVERITY if k in set(data["status"])],
                            range=[theme.STATUS_COLORS[k] for k in _SEVERITY
                                   if k in set(data["status"])]),
            legend=alt.Legend(title="Bewertung", orient="top",
                              labelExpr="datum.label"),
        ),
        opacity=alt.value(0.85),
        tooltip=[alt.Tooltip("Szenario:N"), alt.Tooltip("EV-Anteil:N"),
                 alt.Tooltip("Prozent:Q", format=".0f", title="Spitze (%)"),
                 alt.Tooltip("Bewertung:N")],
    )
    labels = base.mark_text(fontWeight="bold", fontSize=13, color="white").encode(
        text=alt.Text("Prozent:Q", format=".0f"),
    )
    return (cells + labels).properties(width="container", height=42 * max(1, len(order)))


def render_ueberblick(summary: list[dict[str, Any]],
                      timelines: list[dict[str, Any]] | None = None,
                      key: str = "ueberblick") -> None:  # pragma: no cover (UI)
    """Der Überblick-Reiter: das Urteil über alle Szenarien und EV-Anteile."""
    import streamlit as st

    if not summary:
        st.info("Keine Ergebnisse vorhanden — zuerst ein Experiment ausführen.")
        return

    frame = overview_frame(summary, timelines)
    head = headline_numbers(frame)
    if head["status"] is None:
        st.info("Die Ergebnisse enthalten keine Auslastungswerte.")
        return

    # ── Urteil ────────────────────────────────────────────────
    peak_percent = head["worst_peak"] * 100.0
    where = f"{head['worst_scenario']} bei {head['worst_penetration']:.0%} EV-Anteil"
    if head["status"] == "kritisch":
        ceiling = head["safe_ceiling"]
        st.error(
            f"**Das Netz hält nicht durch.** Schlimmster Fall {peak_percent:.0f} % — {where}. "
            + (f"Bis einschließlich {ceiling:.0%} EV-Anteil bleibt jedes Szenario im Rahmen."
               if ceiling is not None else
               "Schon beim niedrigsten geprüften EV-Anteil kommt es zur Überlast.")
        )
    elif head["status"] == "grenzbereich":
        st.warning(f"**Grenzwertig.** Schlimmster Fall {peak_percent:.0f} % — {where}. "
                   "Keine Überschreitung, aber ohne Reserve.")
    else:
        st.success(f"**Das Netz hält durch.** Schlimmster Fall {peak_percent:.0f} % — {where}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Höchste Auslastung", f"{peak_percent:.0f} %", help=where)
    if head["breaking_point"] is not None:
        c2.metric("Überlast ab", f"{head['breaking_point']:.0%} EV",
                  help="Niedrigster EV-Anteil, bei dem irgendein Szenario über 100 % geht")
    else:
        c2.metric("Überlast ab", "—", help="Kein geprüfter EV-Anteil führt zur Überlast")
    c3.metric("Sicher bis", f"{head['safe_ceiling']:.0%} EV" if head["safe_ceiling"] is not None else "—",
              help="Höchster EV-Anteil, bei dem KEIN Szenario über 100 % geht")
    c4.metric("Betroffene Fälle", f"{head['n_over']} / {head['n_total']}",
              help="Kombinationen aus Szenario und EV-Anteil mit Überlast")

    # ── Matrix ────────────────────────────────────────────────
    st.caption(
        "Gilt für das simulierte Netz, nicht für Niederspannungsnetze allgemein: "
        "ein schwach ausgelegtes Testnetz überlastet früh, ein kräftiges hält deutlich "
        "mehr aus. Aussagekräftig ist der **Vergleich der Szenarien untereinander**."
    )

    st.subheader("Szenario × EV-Anteil")
    st.caption(
        "Höchste Auslastung je Kombination, in Prozent — der schlechtere Wert aus "
        "Transformator und Kabel, denn überlastet ist überlastet. Details zu einer "
        "einzelnen Kombination im Reiter **Netzauslastung**."
    )
    st.altair_chart(_matrix_chart(frame), width="stretch")
    download_pair(
        frame.drop(columns=["scenario"]).rename(columns={"status": "Bewertung"}),
        "Überblick", f"{key}_matrix", label="Alle Kombinationen als Tabelle",
    )

    if frame["Kabel"].isna().all():
        st.caption(
            "ℹ️ Dieser Lauf enthält keine gemittelte Kabelspitze (`line_peak_max`) — "
            "gezeigt wird der Transformatorwert. Läufe ab der vereinheitlichten "
            "summary.json enthalten beide."
        )

    # ── Rangfolge beim härtesten Fall ─────────────────────────
    hardest = frame["penetration"].max()
    ranking = scenario_ranking(frame, hardest)
    if not ranking.empty:
        st.subheader(f"Szenarien bei {hardest:.0%} EV-Anteil")
        table = ranking[["Szenario", "Spitze", "curtailment", "soc"]].rename(columns={
            "Spitze": "Höchste Auslastung",
            "curtailment": "§14a-Eingriffe",
            "soc": "EV-Ziel erreicht",
        })
        st.dataframe(
            table.style.format({
                "Höchste Auslastung": lambda v: "—" if pd.isna(v) else f"{v * 100:.0f} %",
                "§14a-Eingriffe": lambda v: "—" if pd.isna(v) else f"{v:.1f}",
                "EV-Ziel erreicht": lambda v: "—" if pd.isna(v) else f"{v:.0%}",
            }),
            width="stretch", hide_index=True,
        )
