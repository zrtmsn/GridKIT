# dashboard/theme.py
# ─────────────────────────────────────────────────────────────
# One place for the dashboard's colours.
#
# Two colour ROLES that must never be mixed:
#   STATUS   : severity of a loading (green→red). Carries meaning: the same
#              value always gets the same colour, in every view.
#   SCENARIO : which policy produced a series. Categorical only; a scenario
#              being orange says nothing about whether it is bad.
#
# The status thresholds are the ones the overload map already used, kept
# identical so the map and the utilisation charts cannot contradict each other.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

# ── Status: loading (p.u.) → severity ────────────────────
# (lower bound, upper bound, key, display label, colour)
#
# The lower bound is EXCLUSIVE, the upper one inclusive: an element at exactly
# 1.0 p.u. is "Grenzbereich", not "Überlast". Not a detail: run_experiment.py
# fills overloaded_lines/-_transformers using `loading > 1.0`, strictly greater.
# Were 1.0 in the red band here, a chart would show red where the overload
# matrix reports nothing, and the two views would contradict each other.
LOAD_BANDS: tuple[tuple[float, float, str, str, str], ...] = (
    (0.0, 0.7, "gut", "unkritisch", "#7cb342"),
    (0.7, 0.9, "warnung", "beobachten", "#ffd54f"),
    (0.9, 1.0, "grenzbereich", "Grenzbereich", "#fdae61"),
    (1.0, float("inf"), "kritisch", "Überlast", "#d7191c"),
)

STATUS_COLORS = {key: color for _, _, key, _, color in LOAD_BANDS}
STATUS_LABELS_DE = {key: label for _, _, key, label, _ in LOAD_BANDS}

#: Above this value an element counts as overloaded (p.u.). The same threshold
#: run_experiment.py uses to fill overloaded_lines/-_transformers.
OVERLOAD_PU = 1.0

#: Shown for a value this run does not contain. Spelled out rather than drawn
#: as a dash: a dash in a column of numbers also reads as a minus sign or as a
#: measured zero, where "keine Angabe" cannot.
NO_VALUE = "k. A."

#: Upper axis bound in percent when the data does not reach higher. Lines hit
#: three-digit values in weak grids, so this is a floor, not a cap.
LOAD_SCALE_MIN_PERCENT = 120.0

# ── Series colours of the utilisation view ──────────────────────
#: Transformer and line are two measurements of the same grid, not scenarios,
#: so they get their own quiet colours instead of the scenario palette.
SERIES_COLORS = {
    "transformer": "#264653",
    "line": "#e76f51",
}
SERIES_LABELS_DE = {
    "transformer": "Trafo-Auslastung",
    "line": "max. Leitungsauslastung",
}

# ── Device types: categorical ──────────────────────────────────
#: Backend key → German display label. The keys are data values
#: (core.constants.DEVICE_*) and stay English.
DEVICE_LABELS_DE = {
    "ev": "EV",
    "battery": "Batterie",
    "hp": "Wärmepumpe",
    "pv": "PV",
}
DEVICE_COLORS = {
    "ev": "#457b9d",
    "battery": "#2a9d8f",
    "hp": "#e63946",
    "pv": "#e9c46a",
}

#: RLlib policy id → device type. One policy per controllable device, shared
#: across every household; hence no PV policy (PV is exogenous, not controllable).
POLICY_DEVICES = {
    "ev_policy": "ev",
    "battery_policy": "battery",
    "hp_policy": "hp",
}


def policy_label(policy_id: str) -> str:
    """German display label for an RLlib policy id ('ev_policy' → 'EV')."""
    return DEVICE_LABELS_DE.get(POLICY_DEVICES.get(policy_id, ""), policy_id)


def policy_color(policy_id: str) -> str:
    """Colour for an RLlib policy id, matching its device type elsewhere."""
    return DEVICE_COLORS.get(POLICY_DEVICES.get(policy_id, ""), "#6c757d")


# ── Scenario: purely categorical ────────────────────────────────
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

#: What each strategy actually does, in one sentence. Kept here rather than in
#: the glossary so a renamed scenario cannot drift apart from its explanation.
#: All four differ ONLY in how the EV charges, except the RL one, where the
#: battery and the heat pump are learned too.
SCENARIO_DESCRIPTIONS_DE = {
    "1: flat / immediate":
        "Das E-Auto lädt mit voller Leistung, sobald es angesteckt ist, bis das "
        "Ladeziel erreicht ist. Kein Blick auf Preis oder Netz. Der Ausgangszustand "
        "ohne jede Steuerung.",
    "2: price-follow (manual)":
        "Das Auto wartet auf das günstigste zusammenhängende Zeitfenster des Tages. "
        "Jeder Haushalt stellt seinen Timer selbst, deshalb streuen die Startzeiten um "
        "dieses Fenster herum statt exakt zusammenzufallen.",
    "2: price-follow (automated)":
        "Dieselbe Regel, aber von einer App gesteuert: alle Haushalte berechnen exakt "
        "dasselbe Fenster und starten gleichzeitig. Genau diese Gleichzeitigkeit kann "
        "eine neue Spitze erzeugen, die höher liegt als ganz ohne Steuerung.",
    "3: selfish RL":
        "Die gelernte Strategie. Jeder Haushalt optimiert nur sich selbst: niedrige "
        "Stromkosten, volles Auto, warme Wohnung. Das Netz kommt in der Belohnung nicht "
        "vor; es wird nur entlastet, weil eine Abregelung den eigenen Nutzen kostet.",
}


def scenario_label(scenario: str) -> str:
    """German display label for a raw scenario key; falls back to the key itself."""
    return SCENARIO_LABELS_DE.get(scenario, scenario)


def scenario_description(scenario: str) -> str:
    """One sentence on what this strategy does; "" for an unknown scenario."""
    return SCENARIO_DESCRIPTIONS_DE.get(scenario, "")


def scenario_short(scenario: str) -> str:
    """Label without its leading number: "konstant / sofort".

    The number groups the strategies (two variants share a 2) and belongs in
    tables, legends and pickers, where the reader is comparing them. Inside a
    sentence it only gets in the way, and after a colon it reads as a second
    colon: "Episode: 1: konstant / sofort".
    """
    return scenario_label(scenario).split(": ", 1)[-1]


def scenario_phrase(scenario: str) -> str:
    """Scenario named inside running text: "Szenario: konstant / sofort"."""
    return f"Szenario: {scenario_short(scenario)}"


def status_of(loading_pu: float) -> str:
    """Severity key for a loading in p.u., the single source of truth for colour.

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
