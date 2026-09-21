# dashboard/controls.py
# ─────────────────────────────────────────────────────────────
# The selection shared across every tab.
#
# Scenario and Ausstattungsgrad are picked on more than one tab. Streamlit refuses to
# reuse a widget key, so each tab needs its own widget, but a user who selects
# "eigennütziges RL bei 60 %" on one tab and finds another tab still showing a
# different scenario has been silently shown two different runs side by side.
# Every picker therefore reads and writes ONE shared value, and each widget
# writes back on change.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any

from dashboard import theme

SCENARIO_STATE = "shared_scenario"
PENETRATION_STATE = "shared_penetration"


def available_scenarios(timelines: list[dict[str, Any]]) -> list[str]:
    """Scenario keys present in the data, in the project's canonical order.

    Anything unrecognised is appended rather than dropped, so a renamed or
    experimental scenario still shows up instead of vanishing silently.
    """
    present = [t.get("scenario") for t in timelines if t.get("scenario")]
    ordered = [s for s in theme.SCENARIO_ORDER if s in present]
    ordered += [s for s in dict.fromkeys(present) if s not in ordered]
    return ordered


def available_penetrations(timelines: list[dict[str, Any]]) -> list[float]:
    """Penetration levels present in the data, ascending."""
    return sorted({float(t.get("penetration", 0.0)) for t in timelines})


def _sync(widget_key: str, shared_key: str) -> None:  # pragma: no cover (UI callback)
    """on_change handler: copy this tab's pick into the state every tab reads."""
    import streamlit as st

    st.session_state[shared_key] = st.session_state[widget_key]


def scenario_picker(timelines: list[dict[str, Any]], key: str,
                    container=None) -> str | None:  # pragma: no cover (UI)
    """Render the shared scenario selector; return the current choice.

    `container` places the widget somewhere other than the current position,
    so a tab can put the selector next to the section it actually changes.
    `key` only namespaces this tab's widget; the value itself lives in session
    state shared by every tab.
    """
    import streamlit as st

    ss = st.session_state
    scenarios = available_scenarios(timelines)
    if not scenarios:
        return None
    if ss.get(SCENARIO_STATE) not in scenarios:
        ss[SCENARIO_STATE] = scenarios[0]

    widget = f"{key}_scenario"
    # Push the shared choice into the widget's own state BEFORE creating it.
    # `index=` is only an initial value: once a widget key holds something,
    # Streamlit uses that and ignores index entirely, which is why syncing
    # through index alone leaves the other tabs on their old selection.
    ss[widget] = ss[SCENARIO_STATE]

    scenario = (container or st).selectbox(
        "Szenario", scenarios,
        format_func=theme.scenario_label,
        key=widget,
        on_change=_sync, args=(widget, SCENARIO_STATE),
        help="Die Regelstrategie der Haushaltsgeräte. Gilt für alle Reiter.",
    )
    ss[SCENARIO_STATE] = scenario
    return scenario


def penetration_picker(timelines: list[dict[str, Any]], key: str,
                       container=None) -> float | None:  # pragma: no cover (UI)
    """Render the shared Ausstattungsgrad selector; return the current choice.

    Renders nothing and simply returns the single value when the run has only
    one device configuration: a run built from one drawn area has nothing to
    pick between, and the selector would be a control that cannot change
    anything.
    """
    import streamlit as st

    ss = st.session_state
    penetrations = available_penetrations(timelines)
    if not penetrations:
        return None
    # default to the hardest case: the story is in what happens under load
    if ss.get(PENETRATION_STATE) not in penetrations:
        ss[PENETRATION_STATE] = penetrations[-1]
    if len(penetrations) == 1:
        ss[PENETRATION_STATE] = penetrations[0]
        return penetrations[0]

    widget = f"{key}_pen"
    ss[widget] = ss[PENETRATION_STATE]
    penetration = (container or st).selectbox(
        "Ausstattungsgrad", penetrations,
        format_func=lambda p: f"{p:.0%}",
        key=widget,
        on_change=_sync, args=(widget, PENETRATION_STATE),
        help=(
            "Anteil der Haushalte mit flexiblen Geräten. Im Batch-Experiment "
            "bekommen genau diese Haushalte die volle Ausstattung (E-Auto, "
            "Batterie, Wärmepumpe und PV), die übrigen keines davon. "
            "Gilt für alle Reiter."
        ),
    )
    ss[PENETRATION_STATE] = penetration
    return penetration


def scenario_penetration_picker(timelines: list[dict[str, Any]], key: str,
                                ) -> tuple[str | None, float | None]:  # pragma: no cover (UI)
    """Both shared selectors, side by side, for tabs where both apply throughout."""
    import streamlit as st

    scenarios = available_scenarios(timelines)
    penetrations = available_penetrations(timelines)
    if not scenarios or not penetrations:
        return None, None
    if len(penetrations) == 1:
        return scenario_picker(timelines, key), penetrations[0]
    left, right = st.columns(2)
    return scenario_picker(timelines, key, left), penetration_picker(timelines, key, right)
