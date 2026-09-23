# dashboard/training.py
# ─────────────────────────────────────────────────────────────
# Training progress: the training tab.
#
# Reads iteration_metrics.json, written by rl_engine.Trainer next to the
# checkpoints. Only four fields are used (training_iteration, episode return
# mean/min/max, `entropy` per policy); RLlib writes ~200 more that are ignored.
#
# The entropy curves are the most informative view here: one shared policy per
# controllable device, so they show which device stops exploring first. A
# policy whose entropy collapses early has committed while the others still
# search.
#
# Everything above the Streamlit section is import-free of streamlit and tested.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard import theme
from dashboard.controls import PENETRATION_STATE
from dashboard.export import download_pair

METRICS_FILENAME = "iteration_metrics.json"


def _sync_penetration(widget_key: str) -> None:  # pragma: no cover (UI callback)
    """Write this tab's pick back to the Ausstattungsgrad the other tabs read."""
    import streamlit as st

    picked = penetration_of(st.session_state[widget_key])
    if picked is not None:
        st.session_state[PENETRATION_STATE] = picked


# ══════════════════════════════════════════════════════════════
# Pure helpers (no Streamlit, unit-tested)
# ══════════════════════════════════════════════════════════════
def penetration_of(directory_name: str) -> float | None:
    """'pen_20' → 0.2, so the training tab can share the other tabs' Ausstattungsgrad."""
    tail = directory_name.removeprefix("pen_")
    try:
        return int(tail) / 100.0
    except ValueError:
        return None


#: Key used when a directory holds the metrics of a single run rather than a sweep.
SINGLE_RUN = "run"


def find_metric_files(output_dir: str | Path) -> dict[str, Path]:
    """{'pen_20': path, …} for every training run with metrics under `output_dir`.

    The two producers lay their output out differently: the sweep writes
    checkpoints/pen_*/iteration_metrics.json, a run from the map writes one
    file at the run root. The latter is keyed SINGLE_RUN, having no penetration
    in its path.

    Missing files are simply absent: an interrupted sweep should still show the
    levels it did train.
    """
    base = Path(output_dir)
    if not base.is_dir():
        return {}

    found: dict[str, Path] = {}
    for directory in sorted((base / "checkpoints").glob("pen_*")):
        candidate = directory / METRICS_FILENAME
        if candidate.is_file():
            found[directory.name] = candidate

    single = base / METRICS_FILENAME
    if not found and single.is_file():
        found[SINGLE_RUN] = single
    return found


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Raw RLlib iteration records; [] when the file is missing or unreadable.

    Returning [] rather than raising because a half-written file from an
    interrupted run should degrade to "no training data" in one tab, not take
    the whole dashboard down.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def return_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Episode return per iteration: mean with the min/max spread."""
    rows = []
    for i, record in enumerate(records, start=1):
        runners = record.get("env_runners") or {}
        mean = runners.get("episode_return_mean")
        if mean is None:
            continue
        rows.append({
            "Iteration": int(record.get("training_iteration", i)),
            "Mittel": float(mean),
            "Minimum": float(runners.get("episode_return_min", mean)),
            "Maximum": float(runners.get("episode_return_max", mean)),
        })
    return pd.DataFrame(rows, columns=["Iteration", "Mittel", "Minimum", "Maximum"])


def entropy_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Long-form policy entropy per iteration, one row per (iteration, policy).

    Only the per-device policies are kept; RLlib also writes an aggregate
    `__all_modules__` entry, which is not a policy and would plot as a fourth
    line with no meaning.
    """
    rows = []
    for i, record in enumerate(records, start=1):
        iteration = int(record.get("training_iteration", i))
        learners = record.get("learners") or {}
        for policy_id, metrics in learners.items():
            if policy_id not in theme.POLICY_DEVICES or not isinstance(metrics, dict):
                continue
            entropy = metrics.get("entropy")
            if entropy is None:
                continue
            rows.append({
                "Iteration": iteration,
                "policy": policy_id,
                "Policy": theme.policy_label(policy_id),
                "Entropie": float(entropy),
            })
    return pd.DataFrame(rows, columns=["Iteration", "policy", "Policy", "Entropie"])


def convergence(records: list[dict[str, Any]]) -> dict[str, Any]:
    """First/last return and the change between them.

    `improved` says only that the return rose over the run; with few
    iterations that is weak evidence, so the view reports the iteration count
    alongside it rather than presenting it as a verdict.
    """
    frame = return_frame(records)
    if frame.empty:
        return {"iterations": 0, "first": None, "last": None, "delta": None, "improved": False}
    first = float(frame["Mittel"].iloc[0])
    last = float(frame["Mittel"].iloc[-1])
    return {
        "iterations": int(len(frame)),
        "first": first,
        "last": last,
        "delta": last - first,
        "improved": last > first,
    }


#: Share of the leading iterations the default view crops away.
WARMUP_FRACTION = 0.25


def convergence_domain(frame: pd.DataFrame,
                       skip_fraction: float = WARMUP_FRACTION) -> tuple[float, float] | None:
    """y-range that shows the converged part, ignoring the first iterations.

    An untrained policy starts catastrophically badly, −13 600 against a
    converged −900 in the reference run, so a full-range axis spends 97 % of
    its height on the warm-up and flattens the part that answers the question.
    The range comes from the tail of the run instead, with a margin.

    None when there is too little to judge: with a handful of iterations the
    whole curve is warm-up and cropping it would hide everything.
    """
    if frame.empty or len(frame) < 8:
        return None
    tail = frame.iloc[int(len(frame) * skip_fraction):]
    if tail.empty:
        return None
    low = float(min(tail["Minimum"].min(), tail["Mittel"].min()))
    high = float(max(tail["Maximum"].max(), tail["Mittel"].max()))
    if low == high:
        return None
    margin = (high - low) * 0.12
    return low - margin, high + margin


def iterations_outside(frame: pd.DataFrame, domain: tuple[float, float] | None) -> int:
    """How many iterations fall outside the zoomed range, for the caption."""
    if domain is None or frame.empty:
        return 0
    low, high = domain
    return int(((frame["Mittel"] < low) | (frame["Mittel"] > high)).sum())


def evaluated_rewards(summary: list[dict[str, Any]],
                      penetration: float | None = None) -> pd.DataFrame:
    """Reward per scenario from summary.json, for comparing the strategies.

    NOT the same quantity as the training return above: the training return is
    summed over every agent in the grid, the evaluated reward is per household,
    a factor of roughly forty apart in the reference run. They must not share an
    axis, which is why this is its own chart rather than a line on the other.
    """
    rows = []
    for record in summary or []:
        if penetration is not None and abs(float(record.get("penetration", -1)) - penetration) > 1e-9:
            continue
        value = record.get("reward_mean")
        if value is None:
            continue
        scenario = record.get("scenario", "")
        rows.append({
            "scenario": scenario,
            "Szenario": theme.scenario_label(scenario),
            "Reward": float(value),
            "Streuung": float(record.get("reward_std") or 0.0),
            "gelernt": scenario == "3: selfish RL",
        })
    frame = pd.DataFrame(rows, columns=["scenario", "Szenario", "Reward", "Streuung", "gelernt"])
    return frame.sort_values("Reward", ascending=False, ignore_index=True)


def final_entropy(records: list[dict[str, Any]]) -> dict[str, float]:
    """{policy_id: entropy at the last iteration it appears in}."""
    frame = entropy_frame(records)
    if frame.empty:
        return {}
    last = frame.sort_values("Iteration").groupby("policy").tail(1)
    return {row["policy"]: float(row["Entropie"]) for _, row in last.iterrows()}


# ══════════════════════════════════════════════════════════════
# Streamlit view
# ══════════════════════════════════════════════════════════════
def _return_chart(frame: pd.DataFrame, domain: tuple[float, float] | None = None):  # pragma: no cover (UI)
    """Mean episode return per iteration, with the min/max spread as a band.

    `domain` crops the y-axis to the converged phase (see convergence_domain).
    """
    import altair as alt

    # clamp so a warm-up iteration far below the range is pinned to the edge
    # rather than silently dropped, so the curve stays continuous
    y_scale = (alt.Scale(domain=list(domain), clamp=True, nice=False)
               if domain else alt.Scale(nice=False, zero=False))
    x_scale = alt.Scale(nice=False, zero=False)

    band = (
        alt.Chart(frame)
        .mark_area(opacity=0.18, color=theme.SERIES_COLORS["transformer"])
        .encode(
            x=alt.X("Iteration:Q", title="Trainingsiteration", scale=x_scale),
            y=alt.Y("Minimum:Q", title="Episoden-Return", scale=y_scale),
            y2="Maximum:Q",
            tooltip=[alt.Tooltip("Iteration:Q"),
                     alt.Tooltip("Minimum:Q", format=".0f"),
                     alt.Tooltip("Maximum:Q", format=".0f")],
        )
    )
    line = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2.4, color=theme.SERIES_COLORS["transformer"], point=True)
        .encode(
            x=alt.X("Iteration:Q", scale=x_scale),
            y=alt.Y("Mittel:Q", title="Episoden-Return", scale=y_scale),
            tooltip=[alt.Tooltip("Iteration:Q"), alt.Tooltip("Mittel:Q", format=".0f")],
        )
    )
    return (band + line).properties(width="container", height=300)


def _reward_chart(frame: pd.DataFrame):  # pragma: no cover (UI)
    """Evaluated reward per strategy, the learned one picked out."""
    import altair as alt

    order = list(frame["Szenario"])
    base = alt.Chart(frame).encode(y=alt.Y("Szenario:N", title=None, sort=order))
    bars = base.mark_bar(height=22, cornerRadiusEnd=3).encode(
        x=alt.X("Reward:Q", title="Reward je Haushalt"),
        # the learned policy in the scenario-3 colour, the baselines muted: the
        # question is where the trained one lands among them
        color=alt.condition(alt.datum.gelernt,
                            alt.value(theme.SCENARIO_COLORS["3: selfish RL"]),
                            alt.value("#9aa7ad")),
        tooltip=[alt.Tooltip("Szenario:N"), alt.Tooltip("Reward:Q", format=".2f"),
                 alt.Tooltip("Streuung:Q", format=".2f", title="± Streuung")],
    )
    spread = base.mark_rule(strokeWidth=1.6, color=theme.SERIES_COLORS["transformer"],
                            opacity=0.8).encode(
        x=alt.X("untere:Q", title=None), x2="obere:Q",
    ).transform_calculate(untere="datum.Reward - datum.Streuung",
                          obere="datum.Reward + datum.Streuung")
    return (bars + spread).properties(width="container", height=alt.Step(32))


def _entropy_chart(frame: pd.DataFrame):  # pragma: no cover (UI)
    """Policy entropy per iteration, one line per device type."""
    import altair as alt

    policies = list(dict.fromkeys(frame["policy"]))
    return (
        alt.Chart(frame)
        .mark_line(strokeWidth=2.2, point=True)
        .encode(
            x=alt.X("Iteration:Q", title="Trainingsiteration",
                    scale=alt.Scale(nice=False, zero=False)),
            y=alt.Y("Entropie:Q", title="Policy-Entropie"),
            color=alt.Color(
                "Policy:N",
                scale=alt.Scale(
                    domain=[theme.policy_label(p) for p in policies],
                    range=[theme.policy_color(p) for p in policies],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[alt.Tooltip("Iteration:Q"), alt.Tooltip("Policy:N"),
                     alt.Tooltip("Entropie:Q", format=".3f")],
        )
        .properties(width="container", height=280)
    )


def render_training(output_dir: str | Path, summary: list[dict[str, Any]] | None = None,
                    key: str = "training") -> None:  # pragma: no cover (UI)
    """The training tab: reward convergence, strategy comparison, entropy per policy."""
    import streamlit as st

    files = find_metric_files(output_dir)
    if not files:
        st.info(
            f"Keine `{METRICS_FILENAME}` gefunden. Sie entsteht beim Training: "
            f"beim Batch-Experiment unter `checkpoints/pen_*/`, bei einem Lauf aus "
            f"der Karte direkt im Lauf-Ordner. Zuerst ein Training ausführen."
        )
        return

    names = list(files)
    if len(names) == 1:
        # Nothing to choose between, so no selector: a run from the map trains
        # one device layout, and a sweep can also have been run over a single
        # Ausstattungsgrad. Testing for SINGLE_RUN alone missed the second case
        # and left a dropdown with one option that could not change anything.
        chosen = names[0]
        if (only := penetration_of(chosen)) is not None:
            st.session_state[PENETRATION_STATE] = only
    else:
        # Share the Ausstattungsgrad with the other tabs: switching to 60 % on the
        # utilisation tab should show the 60 % training run here, not whatever this
        # tab was left on.
        by_penetration = {penetration_of(n): n for n in names if penetration_of(n) is not None}
        shared = st.session_state.get(PENETRATION_STATE)
        widget_key = f"{key}_pen"
        if shared in by_penetration:
            st.session_state[widget_key] = by_penetration[shared]

        chosen = st.selectbox(
            "Ausstattungsgrad", names,
            format_func=lambda n: (f"{p:.0%}" if (p := penetration_of(n)) is not None else n),
            key=widget_key,
            on_change=_sync_penetration, args=(widget_key,),
            help=(
                "Anteil der Haushalte mit flexiblen Geräten. Im Batch-Experiment "
                "bekommen genau diese Haushalte die volle Ausstattung (E-Auto, "
                "Batterie, Wärmepumpe und PV), die übrigen keines davon. "
                "Gilt für alle Reiter."
            ),
        )
        if (picked := penetration_of(chosen)) is not None:
            st.session_state[PENETRATION_STATE] = picked
    records = load_metrics(files[chosen])
    if not records:
        st.warning("Die Metrikdatei ist leer oder nicht lesbar.")
        return

    conv = convergence(records)
    entropies = final_entropy(records)

    c1, c2, c3 = st.columns(3)
    c1.metric("Iterationen", conv["iterations"])
    if conv["last"] is not None:
        c2.metric("Return zuletzt", f"{conv['last']:,.0f}".replace(",", "."),
                  delta=f"{conv['delta']:,.0f}".replace(",", "."),
                  help="Veränderung gegenüber der ersten Iteration")
    if entropies:
        quietest = min(entropies, key=entropies.get)
        c3.metric("Niedrigste Entropie", theme.policy_label(quietest),
                  help=f"{entropies[quietest]:.3f}; diese Policy exploriert am wenigsten")

    st.subheader("Reward-Konvergenz")
    st.caption(
        "Linie: mittlerer Episoden-Return je Iteration. Fläche: Spanne zwischen "
        "schlechtester und bester Episode derselben Iteration. Eine breite Spanne "
        "heißt, die Policy ist noch stark vom Zufall der Episode abhängig."
    )
    returns = return_frame(records)
    domain = convergence_domain(returns)
    if domain is not None:
        zoom = not st.checkbox(
            "Gesamte Spanne inkl. Startphase zeigen", key=f"{key}_full_range",
            help=("Die untrainierte Policy startet um Größenordnungen schlechter. "
                  "Über die volle Spanne gezeichnet ist der konvergierte Teil "
                  "nur noch ein flacher Strich am oberen Rand."),
        )
    else:
        zoom = False
    applied = domain if zoom else None
    st.altair_chart(_return_chart(returns, applied), width="stretch")
    if applied is not None:
        dropped = iterations_outside(returns, applied)
        if dropped:
            st.caption(
                f"Ausschnitt ab der konvergierten Phase: {dropped} frühe "
                f"{'Iteration liegt' if dropped == 1 else 'Iterationen liegen'} "
                f"unterhalb des Ausschnitts und {'ist' if dropped == 1 else 'sind'} "
                "am unteren Rand angeschnitten."
            )
    download_pair(returns, "Reward-Konvergenz", f"{key}_return")

    rewards = evaluated_rewards(summary or [], st.session_state.get(PENETRATION_STATE))
    if not rewards.empty:
        st.subheader("Reward im Vergleich zu den Vergleichsstrategien")
        st.caption(
            "Auswertung des fertigen Laufs: mittlerer Reward je Haushalt, Strich "
            "für die Streuung zwischen den Haushalten. Höher ist besser. Die "
            "gelernte Policy ist hervorgehoben. Achtung, andere Größe als der "
            "Episoden-Return oben: der summiert über alle Agenten im Netz, hier "
            "steht der Wert je Haushalt."
        )
        st.altair_chart(_reward_chart(rewards), width="stretch")
        learned = rewards[rewards["gelernt"]]
        if not learned.empty:
            rank = int(learned.index[0]) + 1
            best = rewards.iloc[0]
            if rank == 1:
                st.caption(
                    f"Die gelernte Policy hat mit {learned.iloc[0]['Reward']:.1f} den "
                    "besten Reward aller Strategien."
                )
            else:
                st.caption(
                    f"Die gelernte Policy liegt auf Platz {rank} von {len(rewards)}; "
                    f"vorn liegt „{best['Szenario']}“ mit {best['Reward']:.1f} "
                    f"gegenüber {learned.iloc[0]['Reward']:.1f}. Der Reward misst "
                    "den Eigennutz der Haushalte, nicht die Netzentlastung; die "
                    "steht im Reiter Überblick."
                )
        download_pair(rewards.drop(columns=["scenario", "gelernt"]),
                      "Reward je Strategie", f"{key}_reward_comparison")

    entropy = entropy_frame(records)
    if not entropy.empty:
        st.subheader("Exploration je Gerätetyp")
        st.caption(
            "Je eine geteilte Policy für EV, Batterie und Wärmepumpe. Fallende "
            "Entropie heißt: die Policy legt sich auf eine Strategie fest. Fällt eine "
            "Kurve deutlich früher als die anderen, hat dieses Gerät sein Verhalten "
            "bereits entschieden, während die übrigen noch suchen."
        )
        st.altair_chart(_entropy_chart(entropy), width="stretch")
        download_pair(entropy, "Entropie", f"{key}_entropie")
