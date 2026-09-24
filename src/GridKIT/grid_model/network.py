#grid_model/network.py
import pypsa
import math
import pandas as pd
from core.models import GridNetwork
import core.constants as const

def build_pypsa_network(grid_network: GridNetwork) -> pypsa.Network:
    """
    Converts a GridNetwork defined in core.models
    into a pypsa network with a slack generator
    """
    network = pypsa.Network(name = grid_network.network_id)
    bus_voltage = {bus.bus_id: bus.v_nom_kv for bus in grid_network.buses}
    network.set_snapshots(pd.date_range(
        # Date is an arbitrary value that has no function, but is required by pypsa
        "2024-01-01",
        periods=const.EPISODE_STEPS,
        freq= f"{const.TIMESTEP_MINUTES}min"))
    
    for bus in grid_network.buses:
        network.add("Bus",bus.bus_id, v_nom = bus.v_nom_kv)
    
    for trafo in grid_network.transformers:
        network.add("Transformer",
                    trafo.trafo_id,
                    bus0 = trafo.hv_bus,
                    bus1 = trafo.lv_bus,
                    s_nom = trafo.s_nom_mva,
                    # per-unit short-circuit impedance — a zero-impedance trafo makes
                    # the power flow singular (nan flows). ~4% uk is typical for LV.
                    x = const.TRANSFORMER_REACTANCE_PU,
                    r = const.TRANSFORMER_RESISTANCE_PU)

    slack_trafo = grid_network.transformers[0]
    network.add(
    "Generator",
    f"slack_{slack_trafo.trafo_id}",
    bus=slack_trafo.hv_bus,
    control="Slack",
    p_nom=1000,
    marginal_cost=1,
    )

    for line in grid_network.lines:
        v_nom = bus_voltage[line.from_bus]
        s_nom = line.max_i_ka * v_nom * math.sqrt(3)
        network.add("Line",
                    line.line_id,
                    bus0 = line.from_bus,
                    bus1 = line.to_bus,
                    x = line.x_ohm_per_km * line.length_km,
                    r = line.r_ohm_per_km * line.length_km,
                    length = line.length_km,
                    s_nom = s_nom)
        
    # NOTE: per-household loads (ev_<bus> and base_<bus>) are added once by
    # GridEnv.__init__ — not here — to avoid duplicate PyPSA load entries.

    return network


