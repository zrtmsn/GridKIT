# dashboard/theme.py
# ─────────────────────────────────────────────────────────────
# One place for the dashboard's colours.
#
# Two colour ROLES that must never be mixed:
#   STATUS   — severity of a loading (green→red). Carries meaning: the same
#              value always gets the same colour, in every view.
#   SCENARIO — which policy produced a series. Categorical only; a scenario
#              being orange says nothing about whether it is bad.
#
# The status thresholds are the ones the overload map already used, kept
# identical so the map and the utilisation charts cannot contradict each other.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

# ── Status: Auslastung (p.u.) → Bewertung ────────────────────
# (untere Grenze, obere Grenze, Schlüssel, Anzeigename, Farbe)
#
# Die untere Grenze ist AUSSCHLIESSEND, die obere einschließend: ein Element mit
# genau 1.0 p.u. ist "Grenzbereich", nicht "Überlast". Das ist kein Detail —
# run_experiment.py füllt overloaded_lines/-_transformers mit `loading > 1.0`,
# also strikt größer. Läge 1.0 hier im roten Band, würde das Diagramm rot zeigen,
# wo die Überlast-Matrix nichts meldet, und beide Ansichten widersprächen sich.
LOAD_BANDS: tuple[tuple[float, float, str, str, str], ...] = (
    (0.0, 0.7, "gut", "unkritisch", "#7cb342"),
    (0.7, 0.9, "warnung", "beobachten", "#ffd54f"),
    (0.9, 1.0, "grenzbereich", "Grenzbereich", "#fdae61"),
    (1.0, float("inf"), "kritisch", "Überlast", "#d7191c"),
)

STATUS_COLORS = {key: color for _, _, key, _, color in LOAD_BANDS}
STATUS_LABELS_DE = {key: label for _, _, key, label, _ in LOAD_BANDS}

#: Über diesem Wert gilt ein Element als überlastet (p.u.). Entspricht der
#: Schwelle, mit der run_experiment.py overloaded_lines/-_transformers füllt.
OVERLOAD_PU = 1.0

#: Achsenobergrenze in Prozent, wenn die Daten nicht höher reichen. Leitungen
#: erreichen in schwachen Netzen dreistellige Werte, daher nur ein Minimum.
LOAD_SCALE_MIN_PERCENT = 120.0

# ── Serienfarben der Auslastungsansicht ──────────────────────
#: Trafo und Leitung sind zwei Messgrößen desselben Netzes, keine Szenarien —
#: daher eigene, ruhige Farben statt der Szenariopalette.
SERIES_COLORS = {
    "transformer": "#264653",
    "line": "#e76f51",
}
SERIES_LABELS_DE = {
    "transformer": "Trafo-Auslastung",
    "line": "max. Leitungsauslastung",
}

# ── Szenario: rein kategorial ────────────────────────────────
SCENARIO_ORDER = [
    "1: flat / immediate",
    "2: price-follow (manual)",
    "2: price-follow (automated)",
    "3: selfish RL",
]
SCENARIO_COLORS = {
    "1: flat / immediate": "#6c757d",
    "2: price-follow (manual)": "#f4a261",
    "2: price-follow (automated)": "#e76f51",
    "3: selfish RL": "#2a9d8f",
}
SCENARIO_LABELS_DE = {
    "1: flat / immediate": "1: konstant / sofort",
    "2: price-follow (manual)": "2: preisorientiert (manuell)",
    "2: price-follow (automated)": "2: preisorientiert (automatisiert)",
    "3: selfish RL": "3: eigennütziges RL",
}


def scenario_label(scenario: str) -> str:
    """German display label for a raw scenario key; falls back to the key itself."""
    return SCENARIO_LABELS_DE.get(scenario, scenario)


def status_of(loading_pu: float) -> str:
    """Severity key for a loading in p.u. — the single source of truth for colour.

    Lower bound exclusive, upper inclusive (see LOAD_BANDS): 1.0 p.u. is
    "grenzbereich", only above it is "kritisch", matching how the backend
    decides what counts as overloaded.
    """
    for lower, _, key, _, _ in reversed(LOAD_BANDS):
        if loading_pu > lower:
            return key
    return LOAD_BANDS[0][2]


def status_color(loading_pu: float) -> str:
    """Colour for a loading in p.u."""
    return STATUS_COLORS[status_of(loading_pu)]
