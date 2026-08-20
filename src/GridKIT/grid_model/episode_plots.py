# grid_model/episode_plots.py
# Shared episode-running + plotting logic used by both
# test_gridcreator_episode.py and cli.py, so the two don't duplicate it.
#
# This is a lightweight dev-tool heuristic for a quick sanity-check plot, not
# an RL baseline — see scenarios/ for the tuned comparison policies.

from pathlib import Path

import matplotlib.pyplot as plt

import core.constants as const
from core.models import BatteryAction, ChargingAction, GridNetwork, HPAction
from grid_model.environment import GridEnv, bus_of, device_of


def _device_soc(env: GridEnv, agent_id: str) -> float:
    hh = env._households[bus_of(agent_id)]
    dev = device_of(agent_id)
    if dev == const.DEVICE_EV:
        return hh.ev.soc
    if dev == const.DEVICE_HEAT_PUMP:
        return hh.hp.thermal_soc
    return hh.battery.soc


def full_when_possible_policy(env: GridEnv) -> dict[str, int]:
    """Simple per-device heuristic for a quick sanity-check episode.

    EV: charge FULL whenever plugged in and below target SoC, else OFF.
    Heat pump: HEAT whenever the thermal buffer is below its comfort floor, else OFF.
    Battery: idle (no arbitrage) — this tool is for a quick network/curtailment
    sanity check, not a tuned baseline.
    """
    actions: dict[str, int] = {}
    for agent_id in env.agent_ids:
        hh = env._households[bus_of(agent_id)]
        dev = device_of(agent_id)
        if dev == const.DEVICE_EV:
            available = hh.ev.is_connected_at(env.current_step)
            needs_charge = hh.ev.soc < hh.ev.target_soc
            actions[agent_id] = int(ChargingAction.FULL if (available and needs_charge) else ChargingAction.OFF)
        elif dev == const.DEVICE_HEAT_PUMP:
            actions[agent_id] = int(HPAction.HEAT if hh.hp.thermal_soc < hh.hp.comfort_min_soc else HPAction.OFF)
        else:  # battery
            actions[agent_id] = int(BatteryAction.IDLE)
    return actions


def run_episode(env: GridEnv, seed: int | None = None) -> dict:
    env.reset(seed=seed)
    log: dict = {
        "transformer_loading_pu": [],
        "max_line_loading_pu": [],
        "curtailment_applied": [],
        "total_curtailed_kw": [],
        "total_load_kw": [],
        "soc": {agent_id: [_device_soc(env, agent_id)] for agent_id in env.agent_ids},
        "reward": {agent_id: [] for agent_id in env.agent_ids},
    }

    for _ in range(const.EPISODE_STEPS):
        actions = full_when_possible_policy(env)
        step_results, power_flow = env.step(actions)

        log["transformer_loading_pu"].append(power_flow.transformer_loading_pu)
        log["max_line_loading_pu"].append(max(power_flow.line_loadings_pu.values(), default=0.0))
        log["curtailment_applied"].append(power_flow.curtailment_applied)
        log["total_curtailed_kw"].append(sum(power_flow.curtailed_power_kw.values()))
        log["total_load_kw"].append(sum(hh.last_net_load_kw for hh in env._households.values()))

        for agent_id in env.agent_ids:
            log["soc"][agent_id].append(_device_soc(env, agent_id))
            log["reward"][agent_id].append(step_results[agent_id].reward)

    return log


def plot_network_topology(network: GridNetwork, out_path: Path) -> None:
    bus_coords = {
        b.bus_id: (b.x_coord, b.y_coord)
        for b in network.buses
        if b.x_coord is not None and b.y_coord is not None
    }
    if not bus_coords:
        print("No bus coordinates available — skipping network topology plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 9))

    for line in network.lines:
        if line.from_bus in bus_coords and line.to_bus in bus_coords:
            x0, y0 = bus_coords[line.from_bus]
            x1, y1 = bus_coords[line.to_bus]
            ax.plot([x0, x1], [y0, y1], color="steelblue", linewidth=0.8, zorder=1)

    for trafo in network.transformers:
        if trafo.hv_bus in bus_coords and trafo.lv_bus in bus_coords:
            x0, y0 = bus_coords[trafo.hv_bus]
            x1, y1 = bus_coords[trafo.lv_bus]
            ax.plot([x0, x1], [y0, y1], color="darkorange", linewidth=2.0, linestyle="--", zorder=1, label="transformer")

    household_ids = set(network.household_bus_ids)
    household_xy = [xy for bus_id, xy in bus_coords.items() if bus_id in household_ids]
    other_xy = [xy for bus_id, xy in bus_coords.items() if bus_id not in household_ids]

    if other_xy:
        ax.scatter(*zip(*other_xy), s=10, color="gray", label="bus", zorder=2)
    if household_xy:
        ax.scatter(*zip(*household_xy), s=14, color="tab:green", label="household", zorder=2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Network topology — {network.network_id} ({len(network.buses)} buses, {network.n_households} households)")
    ax.set_aspect("equal", adjustable="datalim")

    # de-duplicate legend entries (one "transformer" label gets added per transformer otherwise)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_plots(log: dict, network: GridNetwork, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = range(const.EPISODE_STEPS)
    step_label = f"timestep ({const.TIMESTEP_MINUTES} min)"

    plot_network_topology(network, out_dir / "network_topology.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, log["transformer_loading_pu"], label="transformer loading")
    ax.plot(steps, log["max_line_loading_pu"], label="max line loading")
    ax.axhline(1.0, color="red", linestyle="--", label="overload threshold")
    ax.set_xlabel(step_label)
    ax.set_ylabel("loading [p.u.]")
    ax.set_title("Network loading over the episode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "network_loading.png", dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(steps, log["total_load_kw"], color="tab:blue", label="total delivered load")
    ax1.set_xlabel(step_label)
    ax1.set_ylabel("total load [kW]", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.bar(steps, log["total_curtailed_kw"], color="tab:red", alpha=0.4, label="curtailed power")
    ax2.set_ylabel("curtailed power [kW]", color="tab:red")
    for s, applied in zip(steps, log["curtailment_applied"]):
        if applied:
            ax1.axvline(s, color="red", alpha=0.15)
    fig.suptitle("Network load and curtailment events")
    fig.tight_layout()
    fig.savefig(out_dir / "load_and_curtailment.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    soc_matrix = list(log["soc"].values())
    mean_soc = [sum(v) / len(v) for v in zip(*soc_matrix)]
    min_soc = [min(v) for v in zip(*soc_matrix)]
    max_soc = [max(v) for v in zip(*soc_matrix)]
    soc_steps = range(const.EPISODE_STEPS + 1)  # +1: includes the SoC at reset, before any action
    ax.plot(soc_steps, mean_soc, label="mean SoC")
    ax.fill_between(soc_steps, min_soc, max_soc, alpha=0.2, label="min-max range")
    ax.axhline(const.EV_TARGET_SOC, color="green", linestyle="--", label="EV target SoC")
    ax.set_xlabel(step_label)
    ax.set_ylabel("state of charge")
    ax.set_title(f"Device SoC across {len(soc_matrix)} agents (EV / battery / heat pump, mixed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "device_soc.png", dpi=150)
    plt.close(fig)

    total_reward = sum(sum(v) for v in log["reward"].values())
    final_soc = {agent_id: v[-1] for agent_id, v in log["soc"].items()}
    ev_final_soc = {aid: soc for aid, soc in final_soc.items() if device_of(aid) == const.DEVICE_EV}
    n_met_target = sum(1 for soc in ev_final_soc.values() if soc >= const.EV_TARGET_SOC)
    print("Episode summary:")
    print(f"  curtailment events: {sum(log['curtailment_applied'])}/{const.EPISODE_STEPS} timesteps")
    print(f"  peak transformer loading: {max(log['transformer_loading_pu']):.3f} pu")
    print(f"  EV agents reaching target SoC: {n_met_target}/{len(ev_final_soc)}")
    print(f"  total reward across all agents: {total_reward:.2f}")
