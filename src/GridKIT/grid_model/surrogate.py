# grid_model/surrogate.py
# Fast power-flow surrogate for a radial LV feeder.
#
# PyPSA lpf() per step is the training-throughput bottleneck. For a fixed
# radial topology and active-power-only loads, the physics is linear and cheap:
#   - line flow  = sum of loads downstream of that line   (exact, loss-free DC)
#   - trafo flow = sum of all household loads             (exact)
#   - bus voltage ≈ 1 − Σ_path (r·P)/V_base²             (linearized DistFlow)
#
# Validated against PyPSA lpf in test_surrogate.py: flows match exactly,
# voltages within a few 1e-3 pu on the stub feeder.
from __future__ import annotations

import math

from core.models import GridNetwork


class RadialPowerFlow:
    """Linearized solver for a radial feeder rooted at the transformer LV bus."""

    def __init__(self, network: GridNetwork):
        self._network = network
        self._root = network.transformers[0].lv_bus
        self._household_buses = set(network.household_bus_ids)

        self._v_nom = {b.bus_id: b.v_nom_kv for b in network.buses}

        self._line_s_nom: dict[str, float] = {}
        self._line_r_ohm: dict[str, float] = {}
        adjacency: dict[str, list[tuple[str, str]]] = {b.bus_id: [] for b in network.buses}
        for ln in network.lines:
            v = self._v_nom[ln.from_bus]
            self._line_s_nom[ln.line_id] = ln.max_i_ka * v * math.sqrt(3)
            self._line_r_ohm[ln.line_id] = ln.r_ohm_per_km * ln.length_km
            adjacency[ln.from_bus].append((ln.to_bus, ln.line_id))
            adjacency[ln.to_bus].append((ln.from_bus, ln.line_id))

        # BFS from root → parent line of each bus + path of lines root→bus
        self._path_lines: dict[str, list[str]] = {self._root: []}
        order: list[str] = [self._root]
        seen = {self._root}
        i = 0
        while i < len(order):
            bus = order[i]; i += 1
            for nbr, line_id in adjacency[bus]:
                if nbr not in seen:
                    seen.add(nbr)
                    self._path_lines[nbr] = self._path_lines[bus] + [line_id]
                    order.append(nbr)

        # households downstream of each line
        self._downstream: dict[str, set[str]] = {ln.line_id: set() for ln in network.lines}
        for h in self._household_buses:
            for line_id in self._path_lines.get(h, []):
                self._downstream[line_id].add(h)

        self._trafo_s_nom = network.transformers[0].s_nom_mva

    def solve(self, bus_load_mw: dict[str, float]) -> tuple[float, dict[str, float], dict[str, float]]:
        """
        Args:
            bus_load_mw: active power (MW) at each household bus.
        Returns:
            (transformer_loading_pu, {line_id: loading_pu}, {bus_id: voltage_pu})
        """
        line_flow_mw = {
            line_id: sum(bus_load_mw.get(h, 0.0) for h in downstream)
            for line_id, downstream in self._downstream.items()
        }
        line_loadings = {
            line_id: abs(flow) / self._line_s_nom[line_id]
            for line_id, flow in line_flow_mw.items()
        }

        trafo_loading = abs(sum(bus_load_mw.values())) / self._trafo_s_nom

        voltages: dict[str, float] = {}
        for bus_id, v_base in self._v_nom.items():
            drop = sum(
                self._line_r_ohm[lid] * line_flow_mw[lid]
                for lid in self._path_lines.get(bus_id, [])
            ) / (v_base ** 2)
            voltages[bus_id] = 1.0 - drop

        return trafo_loading, line_loadings, voltages
