# grid_model/surrogate.py
# ─────────────────────────────────────────────────────────────
# Fast power-flow surrogate for radial LV grids — one OR MANY feeders.
#
# A real ding0/GridCreator village is a *forest* of LV feeders, each an island
# in the line graph rooted at its MV/LV transformer(s); a feeder may be fed by
# several transformers in parallel (its capacity = their summed kVA). The stub /
# feeder_20 nets are just the single-feeder case.
#
# For a fixed radial topology and active-power-only loads the physics is linear:
#   - line flow  = sum of loads downstream of that line   (exact, loss-free DC)
#   - feeder loading = feeder load / feeder transformer capacity
#   - bus voltage ≈ 1 − Σ_path (r·P)/V_base²             (linearized DistFlow)
#
# `solve()` keeps its 3-tuple signature — the scalar is the MAX feeder loading
# (identical to the transformer loading for a single-feeder net). Per-feeder
# detail (for localized §14a curtailment) is exposed via attributes:
#   household_feeder, household_path_lines, last_feeder_loadings.
#
# Validated against PyPSA pf in test_surrogate.py (single feeder).
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import math

from core.models import GridNetwork


class RadialPowerFlow:
    """Linearized solver for one or many radial LV feeders."""

    def __init__(self, network: GridNetwork):
        self._network = network
        self._v_nom = {b.bus_id: b.v_nom_kv for b in network.buses}

        # line static data + adjacency over lines
        self._line_s_nom: dict[str, float] = {}     # MVA
        self._line_r_ohm: dict[str, float] = {}      # total resistance (Ω)
        adjacency: dict[str, list[tuple[str, str]]] = {b.bus_id: [] for b in network.buses}
        for ln in network.lines:
            v = self._v_nom[ln.from_bus]
            self._line_s_nom[ln.line_id] = ln.max_i_ka * v * math.sqrt(3)
            self._line_r_ohm[ln.line_id] = ln.r_ohm_per_km * ln.length_km
            adjacency[ln.from_bus].append((ln.to_bus, ln.line_id))
            adjacency[ln.to_bus].append((ln.from_bus, ln.line_id))

        # transformer LV buses → summed capacity (parallel transformers add up)
        trafo_cap: dict[str, float] = {}
        for t in network.transformers:
            trafo_cap[t.lv_bus] = trafo_cap.get(t.lv_bus, 0.0) + t.s_nom_mva

        household = set(network.household_bus_ids)

        # connected components over LINES → each becomes an LV feeder iff it
        # contains a transformer LV bus (otherwise it is MV-side, ignored here)
        comp_of: dict[str, int] = {}
        components: list[list[str]] = []
        for start in self._v_nom:
            if start in comp_of:
                continue
            idx = len(components)
            stack = [start]
            comp_of[start] = idx
            members: list[str] = []
            while stack:
                bus = stack.pop()
                members.append(bus)
                for nbr, _ in adjacency[bus]:
                    if nbr not in comp_of:
                        comp_of[nbr] = idx
                        stack.append(nbr)
            components.append(members)

        # per-feeder structure
        self.feeder_capacity: dict[str, float] = {}     # feeder_id → summed trafo kVA (MVA)
        self.feeder_households: dict[str, set[str]] = {}
        self.household_feeder: dict[str, str] = {}
        self.household_path_lines: dict[str, list[str]] = {}
        self._path_lines: dict[str, list[str]] = {}      # bus → root-path lines (all feeder buses)
        self._downstream: dict[str, set[str]] = {ln.line_id: set() for ln in network.lines}

        for members in components:
            trafo_buses = [b for b in members if b in trafo_cap]
            if not trafo_buses:
                continue  # MV-side component, no LV loads to curtail
            root = min(trafo_buses)                       # deterministic feeder root
            feeder_id = root
            self.feeder_capacity[feeder_id] = sum(trafo_cap[b] for b in trafo_buses)
            fh = {b for b in members if b in household}
            self.feeder_households[feeder_id] = fh

            # BFS from the feeder root → path of lines root→bus (within this feeder)
            self._path_lines[root] = []
            seen = {root}
            order = [root]
            i = 0
            while i < len(order):
                bus = order[i]; i += 1
                for nbr, line_id in adjacency[bus]:
                    if nbr not in seen:
                        seen.add(nbr)
                        self._path_lines[nbr] = self._path_lines[bus] + [line_id]
                        order.append(nbr)
            for h in fh:
                self.household_feeder[h] = feeder_id
                path = self._path_lines.get(h, [])
                self.household_path_lines[h] = path
                for line_id in path:
                    self._downstream[line_id].add(h)

        self.last_feeder_loadings: dict[str, float] = {}

    def solve(self, bus_load_mw: dict[str, float]) -> tuple[float, dict[str, float], dict[str, float]]:
        """
        Args:
            bus_load_mw: active power (MW) drawn at each household bus.
        Returns:
            (max_feeder_loading_pu, {line_id: loading_pu}, {bus_id: voltage_pu}).
            Per-feeder loadings are stored on `self.last_feeder_loadings`.
        """
        # line flows = sum of downstream household loads
        line_flow_mw = {
            line_id: sum(bus_load_mw.get(h, 0.0) for h in downstream)
            for line_id, downstream in self._downstream.items()
        }
        line_loadings = {
            line_id: abs(flow) / self._line_s_nom[line_id]
            for line_id, flow in line_flow_mw.items()
        }

        # per-feeder loading = feeder load / feeder transformer capacity
        feeder_loadings: dict[str, float] = {}
        for feeder_id, households in self.feeder_households.items():
            load = sum(bus_load_mw.get(h, 0.0) for h in households)
            cap = self.feeder_capacity[feeder_id]
            feeder_loadings[feeder_id] = abs(load) / cap if cap > 0 else 0.0
        self.last_feeder_loadings = feeder_loadings

        # linearized voltage drop accumulated along the path from each feeder root
        voltages: dict[str, float] = {}
        for bus_id in self._v_nom:
            v_base = self._v_nom[bus_id]
            drop = 0.0
            for line_id in self._path_lines.get(bus_id, []):
                drop += self._line_r_ohm[line_id] * line_flow_mw[line_id] / (v_base ** 2)
            voltages[bus_id] = 1.0 - drop

        max_loading = max(feeder_loadings.values(), default=0.0)
        return max_loading, line_loadings, voltages
