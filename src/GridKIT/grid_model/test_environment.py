# grid_model/test_environment.py
import pytest
import time
from grid_model.environment import GridEnv
from core.models import GridNetwork, Observation, EVState, ChargingAction, StepResult, PowerFlowResult
import core.constants as const
import pytest




@pytest.fixture
def env():
    env = GridEnv()
    env.reset()
    return env

@pytest.fixture
def all_full(env):
    return {agent_id: ChargingAction.FULL for agent_id in env.agent_ids}

@pytest.fixture
def all_off(env):
    return {agent_id: ChargingAction.OFF for agent_id in env.agent_ids}
# ── Constructor tests ─────────────────────────────────────────

def test_env_has_network(env):
    assert isinstance(env.network, GridNetwork)

def test_env_has_correct_agent_ids(env):
    assert env.agent_ids == env.network.household_bus_ids

def test_env_starts_at_step_zero(env):
    assert env.current_step == 0

def test_env_has_ev_for_each_household(env):
    assert len(env._evs) == env.network.n_households

def test_env_pypsa_network_has_ev_loads(env):
    for bus_id in env.network.household_bus_ids:
        assert f"ev_{bus_id}" in env._pypsa_network.loads.index

def test_env_pypsa_network_has_base_loads(env):
    for bus_id in env.network.household_bus_ids:
        assert f"base_{bus_id}" in env._pypsa_network.loads.index


# ── Reset tests ───────────────────────────────────────────────

def test_reset_returns_observations_for_all_agents(env):
    obs = env.reset()
    assert set(obs.keys()) == set(env.agent_ids)

def test_reset_returns_observation_instances(env):
    obs = env.reset()
    for o in obs.values():
        assert isinstance(o, Observation)

def test_reset_resets_step_to_zero(env):
    env._current_step = 10
    env.reset()
    assert env.current_step == 0

def test_reset_samples_soc_in_range(env):
    env.reset(seed=3)
    for ev in env._evs.values():
        assert 0.05 <= ev.soc <= 0.95

def test_reset_ev_loads_are_zero(env):
    env.reset()
    for bus_id in env.network.household_bus_ids:
        assert env._pypsa_network.loads.at[f"ev_{bus_id}", "p_set"] == 0.0

def test_reset_observations_have_correct_soc_progress(env):
    obs = env.reset(seed=5)
    for agent_id, o in obs.items():
        assert o.soc_progress == env._evs[agent_id].soc / env._evs[agent_id].target_soc

def test_reset_with_seed_is_reproducible(env):
    obs1 = env.reset(seed=42)
    obs2 = env.reset(seed=42)
    for agent_id in env.agent_ids:
        assert obs1[agent_id].soc_progress == obs2[agent_id].soc_progress

def test_reset_samples_heterogeneous_schedules(env):
    env.reset(seed=11)
    arrivals = {ev.arrival_step for ev in env._evs.values()}
    # heterogeneity: not every agent shares one identical arrival step
    assert len(arrivals) > 1
    for ev in env._evs.values():
        assert ev.departure_step > ev.arrival_step

def test_different_seeds_give_different_schedules(env):
    env.reset(seed=1)
    a1 = {a: env._evs[a].arrival_step for a in env.agent_ids}
    env.reset(seed=2)
    a2 = {a: env._evs[a].arrival_step for a in env.agent_ids}
    assert a1 != a2

# ── Return types ──────────────────────────────────────────────

def test_step_returns_correct_types(env, all_off):
    step_results, power_flow = env.step(all_off)
    assert isinstance(step_results, dict)
    assert isinstance(power_flow, PowerFlowResult)

def test_step_returns_result_for_each_agent(env, all_off):
    step_results, _ = env.step(all_off)
    assert set(step_results.keys()) == set(env.agent_ids)

def test_step_returns_step_result_instances(env, all_off):
    step_results, _ = env.step(all_off)
    for result in step_results.values():
        assert isinstance(result, StepResult)


# ── Step counter ──────────────────────────────────────────────

def test_step_increments_current_step(env, all_off):
    env.step(all_off)
    assert env.current_step == 1

def test_step_increments_across_multiple_steps(env, all_off):
    for _ in range(5):
        env.step(all_off)
    assert env.current_step == 5


# ── Done flag ─────────────────────────────────────────────────

def test_step_not_done_before_last_step(env, all_off):
    step_results, _ = env.step(all_off)
    for result in step_results.values():
        assert not result.done

def test_step_done_at_last_step(env, all_off):
    for _ in range(const.EPISODE_STEPS - 1):
        env.step(all_off)
    step_results, _ = env.step(all_off)
    for result in step_results.values():
        assert result.done


# ── EV SoC ────────────────────────────────────────────────────

def test_step_off_does_not_charge(env, all_off):
    soc_before = {a: env._evs[a].soc for a in env.agent_ids}
    env.step(all_off)
    for agent_id in env.agent_ids:
        assert env._evs[agent_id].soc == soc_before[agent_id]

def _advance_until_any_connected(env):
    """Step (all OFF) until at least one EV is within its connected window."""
    env.reset(seed=0)
    off = {a: ChargingAction.OFF for a in env.agent_ids}
    for _ in range(const.EPISODE_STEPS):
        connected = [a for a, ev in env._evs.items() if ev.arrival_step <= env.current_step < ev.departure_step]
        if connected:
            return connected
        env.step(off)
    raise AssertionError("no agent ever became connected")

def test_step_full_increases_soc_when_connected(env):
    connected = _advance_until_any_connected(env)
    soc_before = {a: env._evs[a].soc for a in connected}
    env.step({a: ChargingAction.FULL for a in env.agent_ids})
    for a in connected:
        assert env._evs[a].soc > soc_before[a]

def test_disconnected_ev_does_not_charge(env):
    env.reset(seed=0)
    # at step 0 nobody has arrived yet (evening arrival, noon start)
    disconnected = [a for a, ev in env._evs.items() if not (ev.arrival_step <= 0 < ev.departure_step)]
    assert disconnected, "expected some EVs disconnected at step 0"
    soc_before = {a: env._evs[a].soc for a in disconnected}
    env.step({a: ChargingAction.FULL for a in env.agent_ids})
    for a in disconnected:
        assert env._evs[a].soc == soc_before[a]

def test_step_soc_does_not_exceed_1(env, all_full):
    for ev in env._evs.values():
        ev.soc = 1.0
    env.step(all_full)
    for ev in env._evs.values():
        assert ev.soc <= 1.0


# ── Power flow result ─────────────────────────────────────────

def test_power_flow_result_has_correct_timestep(env, all_off):
    _, power_flow = env.step(all_off)
    assert power_flow.timestep == 0

def test_power_flow_result_has_all_buses(env, all_off):
    _, power_flow = env.step(all_off)
    for bus_id in env.network.buses:
        assert bus_id.bus_id in power_flow.bus_voltages_pu

def test_power_flow_result_has_all_lines(env, all_off):
    _, power_flow = env.step(all_off)
    for line in env.network.lines:
        assert line.line_id in power_flow.line_loadings_pu


# ── Curtailment ───────────────────────────────────────────────

def test_no_curtailment_when_not_overloaded(env, all_off):
    _, power_flow = env.step(all_off)
    assert not power_flow.curtailment_applied

#def test_curtailment_caps_ev_power(env, all_full):
#    # force overload by setting threshold very low
#    env._overload_threshold = 0.0
#    env.step(all_full)
#    for agent_id in env.agent_ids:
#        actual_kw = env._pypsa_network.loads.at[f"ev_{agent_id}", "p_set"] * 1000
#        assert actual_kw <= const.MAX_CONTROLLED_POWER_KW


# ── Observations ──────────────────────────────────────────────

def test_step_observation_soc_progress_updates(env, all_full):
    step_results, _ = env.step(all_full)
    for agent_id, result in step_results.items():
        expected = env._evs[agent_id].soc / env._evs[agent_id].target_soc
        assert abs(result.observation.soc_progress - expected) < 1e-6

# ── Observation shape / interface ─────────────────────────────

def test_observation_has_obs_dim_entries(env):
    obs = env.reset(seed=0)
    for o in obs.values():
        assert len(o.to_array()) == const.OBS_DIM

def test_day_ahead_prices_has_one_value_per_step(env):
    env.reset(seed=0)
    assert len(env.day_ahead_prices()) == const.EPISODE_STEPS


# ── Curtailment (proportional + §14a floor) ───────────────────

def _force_all_connected_at(env, step, seed=0):
    env.reset(seed=seed)
    for ev in env._evs.values():
        ev.arrival_step, ev.departure_step = 0, const.EPISODE_STEPS - 1
    off = {a: ChargingAction.OFF for a in env.agent_ids}
    while env.current_step < step:
        env.step(off)

def test_synchronized_full_charging_triggers_curtailment(env):
    _force_all_connected_at(env, 60)
    _, pf = env.step({a: ChargingAction.FULL for a in env.agent_ids})
    assert pf.curtailment_applied
    assert pf.curtailed_power_kw  # someone was dimmed

def test_curtailment_never_below_minimum_guarantee(env):
    _force_all_connected_at(env, 60)
    sr, pf = env.step({a: ChargingAction.FULL for a in env.agent_ids})
    for a in env.agent_ids:
        assert sr[a].info["delivered_kw"] >= const.MIN_GUARANTEED_POWER_KW - 1e-9

def test_no_duplicate_ev_loads(env):
    env.reset(seed=0)
    ev_loads = [idx for idx in env._pypsa_network.loads.index if idx.startswith("ev_")]
    assert len(ev_loads) == len(set(ev_loads)) == env.network.n_households


# ── EV penetration ────────────────────────────────────────────

def test_penetration_limits_agent_count():
    from grid_model.environment import GridEnv
    env = GridEnv(ev_penetration=0.4)   # round(5 * 0.4) = 2
    env.reset(seed=0)
    assert len(env.agent_ids) == 2
    assert len(env._evs) == 2

def test_penetration_full_keeps_all_agents():
    from grid_model.environment import GridEnv
    env = GridEnv(ev_penetration=1.0)
    env.reset(seed=0)
    assert set(env.agent_ids) == set(env.network.household_bus_ids)

def test_non_ev_buses_still_carry_base_load():
    from grid_model.environment import GridEnv
    env = GridEnv(ev_penetration=0.2)   # only 1 EV, but 5 households draw base load
    env.reset(seed=0)
    _, pf = env.step({a: ChargingAction.OFF for a in env.agent_ids})
    assert pf.transformer_loading_pu > 0.0


# ── Benchmark ──────────────────────────────────────────────

def test_step_performance(env, all_full):
    start = time.time()
    env.step(all_full)
    elapsed = time.time() - start
    print(f"\nstep() took {elapsed:.3f}s")
    assert elapsed < 1.0