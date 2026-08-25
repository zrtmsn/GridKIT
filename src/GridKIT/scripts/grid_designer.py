# scripts/grid_designer.py
# ─────────────────────────────────────────────────────────────
# Interactive grid designer (Streamlit + Folium).
#
# Flow:  pick a territory on the map  →  OSM builds a GridNetwork  (or load a
# bundled sample feeder)  →  interactive network graph  →  customise devices by
# %-sliders AND by clicking a household to edit it  →  run the simulation.
#
# Composes: map_ui.osm_fetcher (topology) + core.models (device layout) +
# grid_model.GridEnv (physics) + scenarios (baseline policies). This is an app /
# orchestrator, so it is allowed to import across modules.
#
# UI text is German. const.DEVICE_EV/DEVICE_BATTERY/DEVICE_HEAT_PUMP/DEVICE_PV
# ("ev"/"battery"/"hp"/"pv") are backend data keys — used to look up
# HouseholdDevices.controllable and PowerFlowResult.device_power_kw — and stay
# untranslated. DEVICE_LABELS_DE below is the only place they get a German
# display label, used wherever one of those keys would otherwise be shown
# directly (legend labels, popup text).
#
# Run:  streamlit run src/GridKIT/scripts/grid_designer.py
#       (or: PYTHONPATH=src/GridKIT:src streamlit run .../grid_designer.py)
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── path bootstrap (so `streamlit run` finds the packages) ──
_SCRIPT_DIR = Path(__file__).resolve().parent.parent   # src/GridKIT/
_SRC_DIR = _SCRIPT_DIR.parent                           # src/
for _p in (str(_SCRIPT_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PYTHONPATH", f"{_SCRIPT_DIR}:{_SRC_DIR}")

import numpy as np

import core.constants as const
from core.models import GridNetwork, HouseholdDevices, build_device_layout, device_of

DEVICE_LABELS_DE = {
    const.DEVICE_EV: "EV",
    const.DEVICE_BATTERY: "Batterie",
    const.DEVICE_HEAT_PUMP: "Wärmepumpe",
    const.DEVICE_PV: "PV",
}


# ══════════════════════════════════════════════════════════════
# Pure helpers (no Streamlit — unit-testable)
# ══════════════════════════════════════════════════════════════
_DEFAULT_CENTER = (49.0069, 8.4037)   # Karlsruhe, for coordinate-less sample nets


def bus_coordinates(network: GridNetwork, center=_DEFAULT_CENTER) -> dict[str, tuple[float, float]]:
    """Return {bus_id: (lat, lon)} for every bus.

    Uses real coordinates when present (OSM networks); otherwise synthesizes a
    layout (networkx spring layout if available, else a grid) around `center` so
    a coordinate-less sample feeder can still be drawn and clicked on the map.
    """
    have_coords = [b for b in network.buses if b.x_coord is not None and b.y_coord is not None]
    if len(have_coords) == len(network.buses) and network.buses:
        return {b.bus_id: (float(b.y_coord), float(b.x_coord)) for b in network.buses}

    ids = [b.bus_id for b in network.buses]
    pos: dict[str, tuple[float, float]] = {}
    try:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(ids)
        for ln in network.lines:
            g.add_edge(ln.from_bus, ln.to_bus)
        raw = nx.spring_layout(g, seed=1) if g.number_of_edges() else {b: (0, 0) for b in ids}
        for bid, (x, y) in raw.items():
            pos[bid] = (center[0] + y * 0.010, center[1] + x * 0.014)
    except Exception:
        cols = max(1, int(np.ceil(np.sqrt(len(ids)))))
        for i, bid in enumerate(ids):
            r, c = divmod(i, cols)
            pos[bid] = (center[0] + r * 0.0016, center[1] + c * 0.0022)
    return pos


def nearest_household(lat: float, lon: float, household_coords: dict[str, tuple[float, float]]) -> str | None:
    """Household bus id closest to a clicked (lat, lon)."""
    best, best_d = None, float("inf")
    for bid, (blat, blon) in household_coords.items():
        d = (blat - lat) ** 2 + (blon - lon) ** 2
        if d < best_d:
            best, best_d = bid, d
    return best


def layout_counts(layout: dict[str, HouseholdDevices]) -> dict[str, int]:
    """Count homes equipped with each device type."""
    return {
        const.DEVICE_EV: sum(c.ev for c in layout.values()),
        const.DEVICE_BATTERY: sum(c.battery for c in layout.values()),
        const.DEVICE_HEAT_PUMP: sum(c.heat_pump for c in layout.values()),
        const.DEVICE_PV: sum(c.pv for c in layout.values()),
    }


_DEVICE_COUNT_COLOR = {0: "#adb5bd", 1: "#457b9d", 2: "#e9a13b", 3: "#e63946"}


def household_color(cfg: HouseholdDevices) -> str:
    """Marker colour by number of controllable devices (grey→red)."""
    return _DEVICE_COUNT_COLOR[len(cfg.controllable)]


class StaticBuilder:
    """Adapts an in-memory GridNetwork to the NetworkBuilderProtocol (.build())."""
    def __init__(self, network: GridNetwork):
        self._network = network

    def build(self) -> GridNetwork:
        return self._network


def available_checkpoints(root: str = "outputs/checkpoints") -> list[str]:
    """Names of saved per-penetration RL checkpoints (e.g. ['pen_20','pen_40','pen_60'])."""
    p = Path(root)
    return sorted(d.name for d in p.glob("pen_*") if d.is_dir()) if p.exists() else []


def load_rl_adapter(ckpt_dir: str):  # pragma: no cover (heavy: ray/torch)
    """Load the trained per-device policies as a scenarios.Policy adapter.

    The policies are shared PER DEVICE TYPE over a uniform per-device observation,
    so a policy trained on one feeder transfers to any grid/layout unchanged.
    """
    import os
    from ray.rllib.core.rl_module.rl_module import RLModule
    from GridKIT.rl_engine import RLlibPolicyAdapter
    base = os.path.join(ckpt_dir, "learner_group", "learner", "rl_module")
    modules = {
        d: RLModule.from_checkpoint(os.path.abspath(os.path.join(base, f"{d}_policy")))
        for d in const.CONTROLLABLE_DEVICE_TYPES
    }
    return RLlibPolicyAdapter(modules)


# ══════════════════════════════════════════════════════════════
# Streamlit-App
# ══════════════════════════════════════════════════════════════
def render_designer() -> None:  # pragma: no cover (UI)
    """The designer body (no page config/title) — reusable as a page in the web app."""
    import folium
    import streamlit as st
    from streamlit_folium import st_folium

    from grid_model.builder import StubNetworkBuilder
    from grid_model.environment import GridEnv
    from grid_model.device_profiles import DeviceProfileProvider
    from grid_model.gridcreator_loader import ding0_archive_available
    from scenarios.policies import NaiveImmediatePolicy, NaivePriceFollowPolicy
    from scenarios.runner import run_episode, run_scenario

    ss = st.session_state

    # ── 1. Netzquelle ────────────────────────────────────
    with st.sidebar:
        st.header("1 · Netz")
        source = st.radio("Quelle", [
            "Beispiel-Feeder (20 Haushalte)",
            "Bereich auf der Karte zeichnen",
        ])
        if source.startswith("Beispiel"):
            if st.button("Beispiel-Feeder laden", type="primary"):
                net = StubNetworkBuilder(path="data/feeder_20.json").build()
                _set_network(ss, net)
        else:
            have_ding0 = ding0_archive_available()
            use_ding0 = st.checkbox(
                "Echtes ding0-Netz (GridCreator)", value=have_ding0, disabled=not have_ding0,
                help="Extrahiert das tatsächliche NS-Netz für das gezeichnete Rechteck aus dem "
                     "ding0-Archiv: jeder Transformator real und individuell dimensioniert. OSM "
                     "kann für den ganzen Bereich nur einen generischen 160-kVA-Transformator liefern.",
            )
            if not have_ding0:
                st.caption("⚠️ ding0-Archiv nicht installiert — Rückfall auf OSM, was einen "
                           "**einzelnen synthetischen 160-kVA-Transformator** liefert, sodass "
                           "Engpass-Ergebnisse nicht realistisch sind. Siehe README für den "
                           "Download der input.zip.")
            else:
                st.caption("Rechteck auf der Karte zeichnen, dann erstellen. Nur Deutschland — "
                           "außerhalb der Archivabdeckung fällt der Build auf OSM zurück.")
            if st.button("Aus gezeichnetem Bereich erstellen", type="primary"):
                bounds = ss.get("drawn_bounds")
                if not bounds:
                    st.error("Zuerst ein Rechteck auf der Karte zeichnen.")
                else:
                    _build_from_bounds(st, ss, bounds, use_ding0=use_ding0)

        st.header("2 · Geräte-Mix (%)")
        pen_ev = st.slider("EV", 0, 100, 60) / 100
        pen_bat = st.slider("Batterie", 0, 100, 40) / 100
        pen_hp = st.slider("Wärmepumpe", 0, 100, 40) / 100
        pen_pv = st.slider("PV", 0, 100, 80) / 100
        if st.button("Prozentsätze auf alle Haushalte anwenden"):
            if ss.get("network"):
                homes = list(ss.network.household_bus_ids)
                ss.layout = build_device_layout(homes, ev=pen_ev, battery=pen_bat,
                                                 heat_pump=pen_hp, pv=pen_pv)
                st.success("Layout anhand der Regler aktualisiert.")

    # OSM-Zeichenkarte (nur nötig, solange noch kein Netz existiert)
    if source.startswith("Bereich") and not ss.get("network"):
        _draw_map(st, folium, st_folium, ss)

    if not ss.get("network"):
        st.info("Zum Start ein Netz aus der Seitenleiste laden oder erstellen.")
        return

    network: GridNetwork = ss.network
    coords: dict = ss.bus_coords
    layout: dict = ss.layout
    hh_coords = {b: coords[b] for b in network.household_bus_ids if b in coords}

    counts = layout_counts(layout)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Haushalte", len(network.household_bus_ids))
    c2.metric("EV", counts[const.DEVICE_EV])
    c3.metric("Batterie", counts[const.DEVICE_BATTERY])
    c4.metric("Wärmepumpe", counts[const.DEVICE_HEAT_PUMP])
    c5.metric("PV", counts[const.DEVICE_PV])

    # ── 3. Interaktiver Netzgraph (Haushalt anklicken) ─────
    st.subheader("Netz — Haushalt anklicken, um ihn zu bearbeiten")
    n_grid_nodes = len(network.buses) - len(network.household_bus_ids)
    st.caption(
        f"**Nur die {len(network.household_bus_ids)} farbigen Kreise sind Haushalte, die du bearbeiten kannst.** "
        f"Markerfarbe = Anzahl steuerbarer Geräte: ⬤ grau 0 · ⬤ blau 1 · ⬤ orange 2 · ⬤ rot 3 "
        f"(PV wird im Popup angezeigt). Die blassen grauen Linien + ⚡-Transformator sind das "
        f"elektrische Netzskelett ({n_grid_nodes} Leitungs-/Knotenpunkte) — nur Geometrie, keine "
        f"Haushalte und nicht anklickbar."
    )
    fmap = _network_map(folium, network, coords, layout)
    map_state = st_folium(fmap, height=520, width=None,
                          returned_objects=["last_object_clicked"], key="net_map")
    clicked = (map_state or {}).get("last_object_clicked")
    if clicked and "lat" in clicked:
        sel = nearest_household(clicked["lat"], clicked["lng"], hh_coords)
        if sel:
            ss.selected_bus = sel

    # ── 4. Editor je Haushalt ──────────────────────────────
    sel = ss.get("selected_bus")
    if sel and sel in layout:
        st.subheader(f"Haushalt `{sel}`")
        cfg = layout[sel]
        e1, e2, e3, e4 = st.columns(4)
        ev = e1.checkbox("EV", value=cfg.ev, key=f"ev_{sel}")
        bat = e2.checkbox("Batterie", value=cfg.battery, key=f"bat_{sel}")
        hp = e3.checkbox("Wärmepumpe", value=cfg.heat_pump, key=f"hp_{sel}")
        pv = e4.checkbox("PV", value=cfg.pv, key=f"pv_{sel}")
        layout[sel] = HouseholdDevices(bus_id=sel, ev=ev, battery=bat, heat_pump=hp, pv=pv)
    else:
        st.caption("Kein Haushalt ausgewählt — einen Marker auf der Karte oben anklicken.")

    # ── 5. Simulation ausführen ────────────────────────────────
    st.subheader("3 · Simulation ausführen")
    seeds = st.slider("Evaluierungs-Episoden (Seeds)", 1, 8, 3)
    ckpts = available_checkpoints()
    rcol1, rcol2 = st.columns([1, 1])
    include_rl = rcol1.checkbox("Trainiertes RL einbeziehen (Szenario 3)", value=bool(ckpts), disabled=not ckpts,
                                help="Wendet bereits trainierte Policies je Gerätetyp auf dieses Netz an "
                                     "(kein Training). Trainiert auf dem 20-Haushalte-Feeder → ein Transfer.")
    ckpt_choice = rcol2.selectbox("RL-Checkpoint", ckpts, index=len(ckpts) - 1) if (include_rl and ckpts) else None
    if not ckpts:
        st.caption("Keine trainierten Checkpoints in `outputs/checkpoints/` — zuerst das Batch-Experiment "
                   "ausführen, um RL zu aktivieren.")

    if st.button("Simulation ausführen", type="primary"):
        with st.spinner("Geräteprofile werden erstellt und simuliert…"):
            try:
                provider = DeviceProfileProvider.build()   # cached
                builder = StaticBuilder(network)
                scenarios = [
                    ("1: konstant / sofort", NaiveImmediatePolicy()),
                    ("2: preisorientiert (automatisiert)", NaivePriceFollowPolicy(jitter_std=0.0)),
                ]
                if include_rl and ckpt_choice:
                    if ss.get("_rl_ckpt") != ckpt_choice:
                        ss._rl_adapter = load_rl_adapter(str(Path("outputs/checkpoints") / ckpt_choice))
                        ss._rl_ckpt = ckpt_choice
                    scenarios.append((f"3: eigennütziges RL ({ckpt_choice})", ss._rl_adapter))

                results = {}
                for label, policy in scenarios:
                    env = GridEnv(builder=builder, device_layout=layout, profile_provider=provider)
                    results[label] = run_scenario(env, policy, list(range(seeds)), label=label)
                # eine repräsentative Geräteleistungs-Zeitreihe (sofort)
                env = GridEnv(builder=builder, device_layout=layout, profile_provider=provider)
                rep = run_episode(env, NaiveImmediatePolicy(), 0)
            except Exception as exc:
                st.exception(exc)
                return
        # In session_state ablegen statt sofort zu rendern: sonst verschwindet der
        # ganze Ergebnisblock beim nächsten Rerun (z. B. durch den Darstellung-Regler
        # unten), weil `st.button(...)` dann wieder False ist und dieser Block
        # komplett übersprungen würde.
        ss.sim_results = results
        ss.sim_rep = rep

    if ss.get("sim_results") is not None:
        results = ss.sim_results
        rep = ss.sim_rep

        st.write("**Kennzahlen (Mittelwert ± Std. über die Seeds)**")
        st.table({
            lbl: {
                "Abregelung Ereign./Ep.": f"{s.curtailment_events[0]:.1f}±{s.curtailment_events[1]:.1f}",
                "EV-SoC erreicht": f"{s.soc_satisfaction_rate[0]:.2f}",
                "Trafo-Spitze p.u.": f"{s.transformer_peak_loading_pu[0]:.2f}",
                # das Kabel bindet meist lange vor dem Transformator — nur die Trafo-Zahl
                # zu zeigen liest sich "unauffällig" mitten durch eine thermische Verletzung
                "Leitungs-Spitze p.u.": f"{max((v[0] for v in s.line_peak_loading_pu.values()), default=0.0):.2f}",
                "Leitungen überlastet": len(s.line_overload_steps),
                "Kosten EUR/Tag": f"{s.mean_household_bill_eur[0]:.2f}",
                "Reward": f"{s.mean_episode_reward[0]:.1f}",
            } for lbl, s in results.items()
        })

        st.write("**Wo die Belastung war**")
        for lbl, s in results.items():
            worst = s.worst_feeder()
            bits = []
            if worst and worst[1] > 0:
                bits.append(f"Feeder `{worst[0]}` {worst[1]:.1f} Schritte überlastet (Spitze {worst[2]:.2f} p.u.)")
            top = sorted(s.line_overload_steps.items(), key=lambda kv: -kv[1][0])[:3]
            for lid, (steps, _) in top:
                bits.append(f"Leitung `{lid}` {steps:.1f} Schritte @ {s.line_peak_loading_pu[lid][0]:.2f} p.u.")
            st.caption(f"**{lbl}** — " + ("; ".join(bits) if bits else "nichts über der Nennlast"))

        st.write("**Feeder-Geräteleistung — konstant/sofort**")
        _render_device_power_panels(rep.timestep_results)


# ── Streamlit-Hilfsfunktionen (nur UI) ──────────────────────────────
def _build_from_bounds(st, ss, bounds, *, use_ding0: bool) -> None:  # pragma: no cover (UI)
    """Build the drawn area: real ding0 grid when possible, OSM otherwise.

    ding0 is tried first because it is the only source of real, individually-sized
    transformers. It covers Germany only, and only the districts present in the
    local archive, so any failure degrades to the OSM builder rather than leaving
    the user stuck — but always says so, since the two are not comparable.
    """
    from grid_model.gridcreator_loader import (
        Ding0Unavailable,
        build_grid_network_from_ding0,
    )
    from map_ui.osm_fetcher import OsmFetchConfig, build_grid_network_from_bounds

    if use_ding0:
        try:
            with st.spinner("Das echte ding0-NS-Netz für diesen Bereich wird extrahiert…"):
                net = build_grid_network_from_ding0(
                    south=bounds.south, west=bounds.west,
                    north=bounds.north, east=bounds.east,
                )
            _set_network(ss, net)
            st.success(f"ding0-Netz: {len(net.household_bus_ids)} Haushalte, "
                       f"{len(net.transformers)} echte Transformatoren.")
            return
        except Ding0Unavailable as exc:
            st.warning(f"{exc}\n\nRückfall auf OSM — ein synthetischer 160-kVA-Transformator.")
        except Exception as exc:
            st.warning(f"ding0-Extraktion fehlgeschlagen ({exc}). Rückfall auf OSM.")

    with st.spinner("Overpass wird abgefragt (mit Mirror-Failover) und das Netz wird erstellt…"):
        try:
            res = build_grid_network_from_bounds(bounds, config=OsmFetchConfig())
            _set_network(ss, res.grid_network)
            for w in res.warnings:
                st.warning(w)
        except Exception as exc:
            st.error(f"Erstellung aus OSM fehlgeschlagen — {exc}")


def _set_network(ss, network) -> None:  # pragma: no cover
    ss.network = network
    ss.bus_coords = bus_coordinates(network)
    homes = list(network.household_bus_ids)
    ss.layout = build_device_layout(homes, ev=0.6, battery=0.4, heat_pump=0.4, pv=0.8)
    ss.selected_bus = None
    # verhindert, dass Ergebnisse eines vorherigen Netzes/Layouts weiter angezeigt
    # werden, bis erneut auf "Simulation ausführen" geklickt wird
    ss.sim_results = None
    ss.sim_rep = None


_DEVICE_PANEL_COLORS = {
    const.DEVICE_EV: "#457b9d",
    const.DEVICE_BATTERY: "#2a9d8f",
    const.DEVICE_HEAT_PUMP: "#e63946",
    const.DEVICE_PV: "#e9c46a",
}


_DEVICE_ORDER = (const.DEVICE_EV, const.DEVICE_BATTERY, const.DEVICE_HEAT_PUMP, const.DEVICE_PV)
_CHART_DISPLAYS = ("Linie", "Fläche", "Stufen", "Balken", "Tabelle")


def _apply_mark(chart, display: str):  # pragma: no cover (UI)
    """Same encoded chart, different mark — this is the whole "switch plot type" trick:
    Altair keeps x/y/color/tooltip encodings unchanged, only the mark method differs."""
    if display == "Fläche":
        return chart.mark_area(opacity=0.55, line=True)
    if display == "Stufen":
        return chart.mark_line(interpolate="step-after", strokeWidth=2)
    if display == "Balken":
        return chart.mark_bar()
    return chart.mark_line(strokeWidth=2)  # "Linie" (default)


# ── Schwellenwert-Interpretation ─────────────────────────────────
# Defaults sind bewusst von core.constants abgeleitet statt frei erfunden:
# EV an der §14a-Mindestleistung (der naheliegendste Bezugswert im Projektkontext),
# Batterie/Wärmepumpe an ihrer Nennleistung. PV hat keine feste Nennleistung
# (Anlagengröße wird pro Haushalt aus [PV_PEAK_KWP_MIN, PV_PEAK_KWP_MAX] gezogen),
# daher ein typischer Wert aus der Mitte dieser Spanne.
_DEFAULT_THRESHOLDS_KW = {
    const.DEVICE_EV: const.MIN_GUARANTEED_POWER_KW,
    const.DEVICE_BATTERY: const.BATTERY_MAX_POWER_KW,
    const.DEVICE_HEAT_PUMP: const.HP_RATED_ELECTRIC_KW,
    const.DEVICE_PV: 5.0,
}


def _format_hour(h: float) -> str:
    """"14.25" -> "14:15". `% 24` because the episode runs noon->noon, so h goes past 24."""
    hh = int(h) % 24
    mm = int(round((h - int(h)) * 60)) % 60
    return f"{hh:02d}:{mm:02d}"


def _format_range(start: float, end: float) -> str:
    return _format_hour(start) if abs(start - end) < 1e-9 else f"{_format_hour(start)}–{_format_hour(end)}"


def _contiguous_ranges(hours: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Collapse a boolean mask over `hours` into (start, end) ranges of contiguous True runs —
    so "exceeded from 14:00 to 16:30" instead of listing every single 15-min step."""
    ranges: list[tuple[float, float]] = []
    start_idx = None
    for i, flag in enumerate(mask):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            ranges.append((hours[start_idx], hours[i - 1]))
            start_idx = None
    if start_idx is not None:
        ranges.append((hours[start_idx], hours[-1]))
    return ranges


def _threshold_status(hours: np.ndarray, values: np.ndarray, threshold: float) -> tuple[str, str]:
    """('kritisch'|'warnung'|'gut', message) for one device's day against its threshold.
    Priority matches what the colour is supposed to mean: exceeded (red) beats exactly-at
    (yellow) beats always-below (green) — an exceeded reading is also, technically, "reached
    the threshold", but it should never show as merely a warning."""
    exceeded = values > threshold
    if exceeded.any():
        when = ", ".join(_format_range(a, b) for a, b in _contiguous_ranges(hours, exceeded))
        return "kritisch", f"Schwellenwert ({threshold:.1f} kW) überschritten: {when}"
    equal = np.isclose(values, threshold, atol=0.01)
    if equal.any():
        when = ", ".join(_format_range(a, b) for a, b in _contiguous_ranges(hours, equal))
        return "warnung", f"Schwellenwert ({threshold:.1f} kW) genau erreicht: {when}"
    return "gut", f"Blieb den ganzen Tag unter dem Schwellenwert ({threshold:.1f} kW)."


def _render_device_panel(col, dev: str, df, display: str) -> None:  # pragma: no cover (UI)
    """One device's threshold input + chart (with a dashed threshold line) + the
    resulting interpretation message — all three re-render from the already-computed
    `df`, so moving the number_input needs no simulation re-run, just a normal
    Streamlit rerun (same mechanism the Darstellung selector already relies on)."""
    import altair as alt
    import pandas as pd
    import streamlit as st

    label = DEVICE_LABELS_DE[dev]
    with col:
        threshold = st.number_input(
            f"Schwellenwert {label} (kW)", value=_DEFAULT_THRESHOLDS_KW[dev],
            step=0.5, format="%.1f", key=f"threshold_{dev}",
        )

        chart = _apply_mark(alt.Chart(df, title=label), display).encode(
            x=alt.X("Stunde:Q", title="Stunde"),
            y=alt.Y(f"{dev}:Q", title="kW"),
            color=alt.value(_DEVICE_PANEL_COLORS[dev]),
            tooltip=[alt.Tooltip("Stunde:Q", title="Stunde", format=".2f"),
                     alt.Tooltip(f"{dev}:Q", title=f"{label} (kW)", format=".2f")],
        )
        threshold_rule = (
            alt.Chart(pd.DataFrame({"y": [threshold]}))
            .mark_rule(color="#6c757d", strokeDash=[4, 4], strokeWidth=1.5)
            .encode(y="y:Q")
        )
        st.altair_chart((chart + threshold_rule).properties(width="container", height=220), width="stretch")

        level, message = _threshold_status(df["Stunde"].to_numpy(), df[dev].to_numpy(), threshold)
        icon = {"kritisch": "🔴", "warnung": "🟡", "gut": "🟢"}[level]
        if level == "kritisch":
            st.error(f"{icon} {message}")
        elif level == "warnung":
            st.warning(f"{icon} {message}")
        else:
            st.success(f"{icon} {message}")


def _render_device_power_panels(timestep_results) -> None:  # pragma: no cover (UI)
    """Device-power view for EV/Batterie/Wärmepumpe/PV with a switchable display:
    four individual hoverable charts (line/area/step/bar), a raw value table, or —
    always available except in table mode — one shared chart overlaying all four
    for direct comparison. Altair (unlike matplotlib/st.pyplot) gives hover
    tooltips with the exact value, which is the point of moving off static plots."""
    import altair as alt
    import pandas as pd
    import streamlit as st

    hours = 12 + np.arange(len(timestep_results)) * 0.25
    df = pd.DataFrame({
        "Stunde": hours,
        **{dev: [pf.device_power_kw.get(dev, 0.0) for pf in timestep_results] for dev in _DEVICE_ORDER},
    })

    display = st.radio("Darstellung", _CHART_DISPLAYS, horizontal=True, key="device_power_display")

    export_df = df.rename(columns=DEVICE_LABELS_DE).round(3)
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "📥 Als CSV herunterladen", data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="geraeteleistung.csv", mime="text/csv", key="download_csv_device_power",
    )
    dl2.download_button(
        "📥 Als JSON herunterladen", data=export_df.to_json(orient="records", indent=2, force_ascii=False),
        file_name="geraeteleistung.json", mime="application/json", key="download_json_device_power",
    )

    if display == "Tabelle":
        st.dataframe(export_df, width="stretch", hide_index=True)
        return

    row1 = st.columns(2)
    _render_device_panel(row1[0], const.DEVICE_EV, df, display)
    _render_device_panel(row1[1], const.DEVICE_BATTERY, df, display)
    row2 = st.columns(2)
    _render_device_panel(row2[0], const.DEVICE_HEAT_PUMP, df, display)
    _render_device_panel(row2[1], const.DEVICE_PV, df, display)

    st.write("**Vergleich aller Geräte**")
    long_df = df.melt(id_vars="Stunde", var_name="dev", value_name="kW")
    long_df["Gerät"] = long_df["dev"].map(DEVICE_LABELS_DE)
    domain = [DEVICE_LABELS_DE[d] for d in _DEVICE_ORDER]
    color_range = [_DEVICE_PANEL_COLORS[d] for d in _DEVICE_ORDER]
    shared = (
        _apply_mark(alt.Chart(long_df, title="Alle Geräte im Vergleich"), display)
        .encode(
            x=alt.X("Stunde:Q", title="Stunde"),
            y=alt.Y("kW:Q", title="kW"),
            color=alt.Color("Gerät:N", scale=alt.Scale(domain=domain, range=color_range), legend=alt.Legend(title="Gerät")),
            tooltip=[alt.Tooltip("Stunde:Q", title="Stunde", format=".2f"),
                     alt.Tooltip("Gerät:N", title="Gerät"),
                     alt.Tooltip("kW:Q", title="kW", format=".2f")],
        )
        .properties(width="container", height=320)
    )
    st.altair_chart(shared, width="stretch")


def _draw_map(st, folium, st_folium, ss) -> None:  # pragma: no cover
    from folium.plugins import Draw
    m = folium.Map(location=_DEFAULT_CENTER, zoom_start=14, tiles="OpenStreetMap")
    Draw(export=False, draw_options={"polyline": False, "circle": False, "circlemarker": False,
                                     "marker": False, "rectangle": True, "polygon": False}).add_to(m)
    data = st_folium(m, height=760, width=None, use_container_width=True,
                     returned_objects=["last_active_drawing"], key="draw_map")
    from map_ui.osm_fetcher import AreaBounds
    drawing = (data or {}).get("last_active_drawing")
    if drawing and drawing.get("geometry", {}).get("type") == "Polygon":
        pts = drawing["geometry"]["coordinates"][0]
        lons = [p[0] for p in pts]; lats = [p[1] for p in pts]
        ss.drawn_bounds = AreaBounds(south=min(lats), west=min(lons), north=max(lats), east=max(lons))
        st.success(f"Bereich ausgewählt (~{ss.drawn_bounds.approx_area_km2():.2f} km²). "
                   "»Aus gezeichnetem Bereich erstellen« in der Seitenleiste anklicken.")


def _network_map(folium, network, coords, layout):  # pragma: no cover
    lats = [coords[b][0] for b in coords]; lons = [coords[b][1] for b in coords]
    center = (float(np.mean(lats)), float(np.mean(lons))) if coords else _DEFAULT_CENTER
    m = folium.Map(location=center, zoom_start=15, tiles="cartodbpositron")

    # das elektrische Skelett (Leitungen) als blasse Linien — nur Geometrie, nicht anklickbar
    for ln in network.lines:
        a, b = coords.get(ln.from_bus), coords.get(ln.to_bus)
        if a and b:
            folium.PolyLine([a, b], color="#c3c9d1", weight=1.2, opacity=0.6).add_to(m)

    # Transformator(en): der Netzkopf — ein eigenes Symbol, kein Haushalt
    trafo_buses = {t.hv_bus for t in network.transformers} | {t.lv_bus for t in network.transformers}
    for tb in trafo_buses:
        if tb in coords:
            folium.Marker(coords[tb], icon=folium.Icon(color="black", icon="bolt", prefix="fa"),
                          tooltip=f"Transformator · {tb}").add_to(m)

    # NUR Haushalte sind Marker, die man anklicken & bearbeiten kann (alles andere bleibt eine Linie)
    for hb in network.household_bus_ids:
        if hb not in coords:
            continue
        cfg = layout.get(hb)
        devs = ", ".join(DEVICE_LABELS_DE.get(d, d) for d in cfg.controllable) if cfg and cfg.controllable else "keine"
        popup = f"{hb} — Geräte: {devs}{' + PV' if cfg and cfg.pv else ''}"
        folium.CircleMarker(coords[hb], radius=6, color=household_color(cfg) if cfg else "#adb5bd",
                            fill=True, fill_opacity=0.9, weight=2, popup=popup,
                            tooltip="zum Bearbeiten dieses Haushalts anklicken").add_to(m)
    return m


def main() -> None:  # pragma: no cover (UI)
    import streamlit as st
    st.set_page_config(page_title="GridKIT — Netzentwurf", layout="wide")
    st.title("GridKIT — Netzentwurf")
    st.caption("Bereich wählen → Netz erstellen → Geräte anpassen (Regler + Haushalt anklicken) → Simulation ausführen.")
    render_designer()


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    main()
elif _running_under_streamlit():
    # `streamlit run` imports the module (name != __main__); auto-run only then,
    # NOT on a plain import (e.g. pytest importing the pure helpers).
    main()
