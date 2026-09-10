# dashboard/training.py
# ─────────────────────────────────────────────────────────────
# Trainingsverlauf — der Training-Reiter.
#
# Reads iteration_metrics.json, written per penetration level by
# rl_engine.Trainer into its checkpoint directory:
#   outputs/checkpoints/pen_{20,40,60}/iteration_metrics.json
#
# Scoped to the fields the dashboard agreed on: training_iteration,
# episode_return_mean/min/max, and `entropy` per policy. RLlib writes ~200
# more (timers, config, per-optimiser state) that are deliberately ignored.
#
# The three entropy curves are the most informative single view here: one
# shared policy per controllable device (EV, Batterie, Wärmepumpe), so their
# curves show which device stops exploring first — a policy whose entropy
# collapses early has committed to a strategy while the others are still
# searching.
#
# Everything above the Streamlit section is import-free of streamlit and
# unit-tested.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard import theme

METRICS_FILENAME = "iteration_metrics.json"


# ══════════════════════════════════════════════════════════════
# Reine Helfer (kein Streamlit — unit-testbar)
# ══════════════════════════════════════════════════════════════
def find_metric_files(checkpoints_dir: str | Path) -> dict[str, Path]:
    """{'pen_20': path, …} for every penetration that has metrics on disk.

    Missing files are simply absent from the result: a sweep can be interrupted
    part-way, and the ones already trained should still be viewable.
    """
    base = Path(checkpoints_dir)
    if not base.is_dir():
        return {}
    found = {}
    for directory in sorted(base.glob("pen_*")):
        candidate = directory / METRICS_FILENAME
        if candidate.is_file():
            found[directory.name] = candidate
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

    `improved` says only that the return rose over the run — with few
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


def final_entropy(records: list[dict[str, Any]]) -> dict[str, float]:
    """{policy_id: entropy at the last iteration it appears in}."""
    frame = entropy_frame(records)
    if frame.empty:
        return {}
    last = frame.sort_values("Iteration").groupby("policy").tail(1)
    return {row["policy"]: float(row["Entropie"]) for _, row in last.iterrows()}


# ══════════════════════════════════════════════════════════════
# Streamlit-Ansicht
# ══════════════════════════════════════════════════════════════
def _return_chart(frame: pd.DataFrame):  # pragma: no cover (UI)
    import altair as alt

    band = (
        alt.Chart(frame)
        .mark_area(opacity=0.18, color=theme.SERIES_COLORS["transformer"])
        .encode(
            x=alt.X("Iteration:Q", title="Trainingsiteration",
                    scale=alt.Scale(nice=False, zero=False)),
            y=alt.Y("Minimum:Q", title="Episoden-Return"),
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
            x=alt.X("Iteration:Q", scale=alt.Scale(nice=False, zero=False)),
            y=alt.Y("Mittel:Q", title="Episoden-Return"),
            tooltip=[alt.Tooltip("Iteration:Q"), alt.Tooltip("Mittel:Q", format=".0f")],
        )
    )
    return (band + line).properties(width="container", height=300)


def _entropy_chart(frame: pd.DataFrame):  # pragma: no cover (UI)
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


def render_training(checkpoints_dir: str | Path, key: str = "training") -> None:  # pragma: no cover (UI)
    """Der Training-Reiter: Reward-Konvergenz und Entropie je Policy."""
    import streamlit as st

    files = find_metric_files(checkpoints_dir)
    if not files:
        st.info(
            f"Keine `{METRICS_FILENAME}` gefunden. Sie entsteht beim Training unter "
            "`checkpoints/pen_*/` — zuerst ein Experiment ausführen."
        )
        return

    names = list(files)
    chosen = st.selectbox("EV-Anteil", names,
                          format_func=lambda n: f"{n.removeprefix('pen_')} %",
                          key=f"{key}_pen")
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
                  help=f"{entropies[quietest]:.3f} — diese Policy exploriert am wenigsten")

    if conv["iterations"] < 10:
        st.warning(
            f"Nur {conv['iterations']} Iterationen — zu wenig, um aus dem Verlauf auf "
            "die Qualität der Policy zu schließen. Für belastbare Aussagen mit den "
            "voreingestellten 40 Iterationen trainieren."
        )

    st.subheader("Reward-Konvergenz")
    st.caption(
        "Linie: mittlerer Episoden-Return je Iteration. Fläche: Spanne zwischen "
        "schlechtester und bester Episode derselben Iteration — eine breite Spanne "
        "heißt, die Policy ist noch stark vom Zufall der Episode abhängig."
    )
    st.altair_chart(_return_chart(return_frame(records)), width="stretch")

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
