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
# Streamlit app
# ══════════════════════════════════════════════════════════════
def render_designer() -> None:  # pragma: no cover (UI)
    """The designer body (no page config/title) — reusable as a page in the web app."""
    import folium
    import matplotlib.pyplot as plt
    import streamlit as st
    from streamlit_folium import st_folium

    from grid_model.builder import StubNetworkBuilder
    from grid_model.environment import GridEnv
    from grid_model.device_profiles import DeviceProfileProvider
    from grid_model.gridcreator_loader import ding0_archive_available
    from scenarios.policies import NaiveImmediatePolicy, NaivePriceFollowPolicy
    from scenarios.runner import run_episode, run_scenario

    ss = st.session_state

    # ── 1. Network source ────────────────────────────────────
    with st.sidebar:
        st.header("1 · Network")
        source = st.radio("Source", [
            "Sample feeder (20 homes)",
            "Draw area on map",
        ])
        if source.startswith("Sample"):
            if st.button("Load sample feeder", type="primary"):
                net = StubNetworkBuilder(path="data/feeder_20.json").build()
                _set_network(ss, net)
        else:
            have_ding0 = ding0_archive_available()
            use_ding0 = st.checkbox(
                "Real ding0 grid (GridCreator)", value=have_ding0, disabled=not have_ding0,
                help="Extracts the actual LV grid for the drawn box from the ding0 archive: every "
                     "transformer real and individually sized. OSM can only ever give one "
                     "generic 160 kVA transformer for the whole area.",
            )
            if not have_ding0:
                st.caption("⚠️ ding0 archive not installed — falling back to OSM, which gives a "
                           "**single synthetic 160 kVA transformer**, so congestion results are "
                           "not realistic. See the README for the input.zip download.")
            else:
                st.caption("Draw a rectangle on the map, then build. Germany only — outside the "
                           "archive's coverage the build falls back to OSM.")
            if st.button("Build from drawn area", type="primary"):
                bounds = ss.get("drawn_bounds")
                if not bounds:
                    st.error("Draw a rectangle on the map first.")
                else:
                    _build_from_bounds(st, ss, bounds, use_ding0=use_ding0)

        st.header("2 · Device mix (%)")
        pen_ev = st.slider("EV", 0, 100, 60) / 100
        pen_bat = st.slider("Battery", 0, 100, 40) / 100
        pen_hp = st.slider("Heat pump", 0, 100, 40) / 100
        pen_pv = st.slider("PV", 0, 100, 80) / 100
        if st.button("Apply %s to all households"):
            if ss.get("network"):
                homes = list(ss.network.household_bus_ids)
                ss.layout = build_device_layout(homes, ev=pen_ev, battery=pen_bat,
                                                 heat_pump=pen_hp, pv=pen_pv)
                st.success("Layout updated from sliders.")

    # OSM draw map (only needed before a network exists)
    if source.startswith("Draw") and not ss.get("network"):
        _draw_map(st, folium, st_folium, ss)

    if not ss.get("network"):
        st.info("Load or build a network from the sidebar to start.")
        return

    network: GridNetwork = ss.network
    coords: dict = ss.bus_coords
    layout: dict = ss.layout
    hh_coords = {b: coords[b] for b in network.household_bus_ids if b in coords}

    counts = layout_counts(layout)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Households", len(network.household_bus_ids))
    c2.metric("EV", counts[const.DEVICE_EV])
    c3.metric("Battery", counts[const.DEVICE_BATTERY])
    c4.metric("Heat pump", counts[const.DEVICE_HEAT_PUMP])
    c5.metric("PV", counts[const.DEVICE_PV])

    # ── 3. Interactive network graph (click a household) ─────
    st.subheader("Network — click a household to edit it")
    n_grid_nodes = len(network.buses) - len(network.household_bus_ids)
    st.caption(
        f"**Only the {len(network.household_bus_ids)} coloured circles are households you can edit.** "
        f"Marker colour = number of controllable devices: ⬤ grey 0 · ⬤ blue 1 · ⬤ orange 2 · ⬤ red 3 "
        f"(PV shown in the popup). The faint grey lines + ⚡ transformer are the electrical grid skeleton "
        f"({n_grid_nodes} wire/junction nodes) — geometry only, not homes and not clickable."
    )
    fmap = _network_map(folium, network, coords, layout)
    map_state = st_folium(fmap, height=520, width=None,
                          returned_objects=["last_object_clicked"], key="net_map")
    clicked = (map_state or {}).get("last_object_clicked")
    if clicked and "lat" in clicked:
        sel = nearest_household(clicked["lat"], clicked["lng"], hh_coords)
        if sel:
            ss.selected_bus = sel

    # ── 4. Per-household editor ──────────────────────────────
    sel = ss.get("selected_bus")
    if sel and sel in layout:
        st.subheader(f"Household `{sel}`")
        cfg = layout[sel]
        e1, e2, e3, e4 = st.columns(4)
        ev = e1.checkbox("EV", value=cfg.ev, key=f"ev_{sel}")
        bat = e2.checkbox("Battery", value=cfg.battery, key=f"bat_{sel}")
        hp = e3.checkbox("Heat pump", value=cfg.heat_pump, key=f"hp_{sel}")
        pv = e4.checkbox("PV", value=cfg.pv, key=f"pv_{sel}")
        layout[sel] = HouseholdDevices(bus_id=sel, ev=ev, battery=bat, heat_pump=hp, pv=pv)
    else:
        st.caption("No household selected — click a marker on the map above.")

    # ── 5. Run the simulation ────────────────────────────────
    st.subheader("3 · Run simulation")
    seeds = st.slider("Evaluation episodes (seeds)", 1, 8, 3)
    ckpts = available_checkpoints()
    rcol1, rcol2 = st.columns([1, 1])
    include_rl = rcol1.checkbox("Include trained RL (scenario 3)", value=bool(ckpts), disabled=not ckpts,
                                help="Applies already-trained per-device policies to this grid (no training). "
                                     "Trained on the 20-home feeder → a transfer.")
    ckpt_choice = rcol2.selectbox("RL checkpoint", ckpts, index=len(ckpts) - 1) if (include_rl and ckpts) else None
    if not ckpts:
        st.caption("No trained checkpoints in `outputs/checkpoints/` — run the batch experiment first to enable RL.")

    if st.button("Run simulation", type="primary"):
        with st.spinner("Building device profiles and simulating…"):
            try:
                provider = DeviceProfileProvider.build()   # cached
                builder = StaticBuilder(network)
                scenarios = [
                    ("1: flat / immediate", NaiveImmediatePolicy()),
                    ("2: price-follow (automated)", NaivePriceFollowPolicy(jitter_std=0.0)),
                ]
                if include_rl and ckpt_choice:
                    if ss.get("_rl_ckpt") != ckpt_choice:
                        ss._rl_adapter = load_rl_adapter(str(Path("outputs/checkpoints") / ckpt_choice))
                        ss._rl_ckpt = ckpt_choice
                    scenarios.append((f"3: selfish RL ({ckpt_choice})", ss._rl_adapter))

                results = {}
                for label, policy in scenarios:
                    env = GridEnv(builder=builder, device_layout=layout, profile_provider=provider)
                    results[label] = run_scenario(env, policy, list(range(seeds)), label=label)
                # a representative device-power timeline (immediate)
                env = GridEnv(builder=builder, device_layout=layout, profile_provider=provider)
                rep = run_episode(env, NaiveImmediatePolicy(), 0)
            except Exception as exc:
                st.exception(exc)
                return

        st.write("**Metrics (mean ± std over seeds)**")
        st.table({
            lbl: {
                "curtailment ev/ep": f"{s.curtailment_events[0]:.1f}±{s.curtailment_events[1]:.1f}",
                "EV SoC ok": f"{s.soc_satisfaction_rate[0]:.2f}",
                "trafo peak pu": f"{s.transformer_peak_loading_pu[0]:.2f}",
                # the cable usually binds long before the transformer does — showing only
                # the trafo figure reads "comfortable" straight through a thermal violation
                "line peak pu": f"{max((v[0] for v in s.line_peak_loading_pu.values()), default=0.0):.2f}",
                "lines overloaded": len(s.line_overload_steps),
                "bill EUR/day": f"{s.mean_household_bill_eur[0]:.2f}",
                "reward": f"{s.mean_episode_reward[0]:.1f}",
            } for lbl, s in results.items()
        })

        st.write("**Where the stress was**")
        for lbl, s in results.items():
            worst = s.worst_feeder()
            bits = []
            if worst and worst[1] > 0:
                bits.append(f"feeder `{worst[0]}` overloaded {worst[1]:.1f} steps (peak {worst[2]:.2f} pu)")
            top = sorted(s.line_overload_steps.items(), key=lambda kv: -kv[1][0])[:3]
            for lid, (steps, _) in top:
                bits.append(f"line `{lid}` {steps:.1f} steps @ {s.line_peak_loading_pu[lid][0]:.2f} pu")
            st.caption(f"**{lbl}** — " + ("; ".join(bits) if bits else "nothing exceeded its rating"))

        tr = rep.timestep_results
        hours = 12 + np.arange(len(tr)) * 0.25
        fig, ax = plt.subplots(figsize=(10, 3.2))
        for dev, color in ((const.DEVICE_EV, "#457b9d"), (const.DEVICE_HEAT_PUMP, "#e63946"),
                           (const.DEVICE_BATTERY, "#2a9d8f")):
            ax.plot(hours, [pf.device_power_kw.get(dev, 0.0) for pf in tr], label=dev, color=color)
        ax.plot(hours, [-pf.device_power_kw.get(const.DEVICE_PV, 0.0) for pf in tr],
                label="pv (−gen)", color="#e9c46a", ls="--", alpha=0.7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("Feeder device power — flat/immediate")
        ax.set_ylabel("kW"); ax.set_xlabel("hour"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
        st.pyplot(fig)


# ── Streamlit helpers (UI-only) ──────────────────────────────
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
            with st.spinner("Extracting the real ding0 LV grid for this area…"):
                net = build_grid_network_from_ding0(
                    south=bounds.south, west=bounds.west,
                    north=bounds.north, east=bounds.east,
                )
            _set_network(ss, net)
            st.success(f"ding0 grid: {len(net.household_bus_ids)} homes, "
                       f"{len(net.transformers)} real transformers.")
            return
        except Ding0Unavailable as exc:
            st.warning(f"{exc}\n\nFalling back to OSM — one synthetic 160 kVA transformer.")
        except Exception as exc:
            st.warning(f"ding0 extraction failed ({exc}). Falling back to OSM.")

    with st.spinner("Querying Overpass (with mirror failover) and building the grid…"):
        try:
            res = build_grid_network_from_bounds(bounds, config=OsmFetchConfig())
            _set_network(ss, res.grid_network)
            for w in res.warnings:
                st.warning(w)
        except Exception as exc:
            st.error(f"Couldn't build from OSM — {exc}")


def _set_network(ss, network) -> None:  # pragma: no cover
    ss.network = network
    ss.bus_coords = bus_coordinates(network)
    homes = list(network.household_bus_ids)
    ss.layout = build_device_layout(homes, ev=0.6, battery=0.4, heat_pump=0.4, pv=0.8)
    ss.selected_bus = None


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
        st.success(f"Area selected (~{ss.drawn_bounds.approx_area_km2():.2f} km²). Click 'Build' in the sidebar.")


def _network_map(folium, network, coords, layout):  # pragma: no cover
    lats = [coords[b][0] for b in coords]; lons = [coords[b][1] for b in coords]
    center = (float(np.mean(lats)), float(np.mean(lons))) if coords else _DEFAULT_CENTER
    m = folium.Map(location=center, zoom_start=15, tiles="cartodbpositron")

    # the electrical skeleton (wires) as faint lines — geometry only, not clickable
    for ln in network.lines:
        a, b = coords.get(ln.from_bus), coords.get(ln.to_bus)
        if a and b:
            folium.PolyLine([a, b], color="#c3c9d1", weight=1.2, opacity=0.6).add_to(m)

    # transformer(s): the grid head — a distinct icon, not a household
    trafo_buses = {t.hv_bus for t in network.transformers} | {t.lv_bus for t in network.transformers}
    for tb in trafo_buses:
        if tb in coords:
            folium.Marker(coords[tb], icon=folium.Icon(color="black", icon="bolt", prefix="fa"),
                          tooltip=f"transformer · {tb}").add_to(m)

    # ONLY households are markers you can click & edit (everything else stays a line)
    for hb in network.household_bus_ids:
        if hb not in coords:
            continue
        cfg = layout.get(hb)
        devs = ", ".join(cfg.controllable) if cfg and cfg.controllable else "none"
        popup = f"{hb} — devices: {devs}{' + PV' if cfg and cfg.pv else ''}"
        folium.CircleMarker(coords[hb], radius=6, color=household_color(cfg) if cfg else "#adb5bd",
                            fill=True, fill_opacity=0.9, weight=2, popup=popup,
                            tooltip="click to edit this household").add_to(m)
    return m


def main() -> None:  # pragma: no cover (UI)
    import streamlit as st
    st.set_page_config(page_title="GridKIT — Grid Designer", layout="wide")
    st.title("GridKIT — Grid Designer")
    st.caption("Pick an area → build a grid → customise devices (sliders + click a household) → run the sim.")
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
