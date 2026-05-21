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

def test_reset_resets_ev_soc(env):
    for ev in env._evs.values():
        ev.soc = 0.9
    env.reset()
    for ev in env._evs.values():
        assert ev.soc == const.EV_INITIAL_SOC_MEAN

def test_reset_ev_loads_are_zero(env):
    env.reset()
    for bus_id in env.network.household_bus_ids:
        assert env._pypsa_network.loads.at[f"ev_{bus_id}", "p_set"] == 0.0

def test_reset_observations_have_correct_soc_progress(env):
    obs = env.reset()
    for agent_id, o in obs.items():
        assert o.soc_progress == const.EV_INITIAL_SOC_MEAN

def test_reset_with_seed_is_reproducible(env):
    obs1 = env.reset(seed=42)
    obs2 = env.reset(seed=42)
    for agent_id in env.agent_ids:
        assert obs1[agent_id].soc_progress == obs2[agent_id].soc_progress

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

def test_step_full_increases_soc(env, all_full):
    soc_before = {a: env._evs[a].soc for a in env.agent_ids}
    env.step(all_full)
    for agent_id in env.agent_ids:
        assert env._evs[agent_id].soc > soc_before[agent_id]

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

# ── Benchmark ──────────────────────────────────────────────

def test_step_performance(env, all_full):    
    start = time.time()
    env.reset()
    for i in range(const.EPISODE_STEPS):
        env.step(all_full)
    elapsed = time.time() - start
    print(f"\none episode took {elapsed:.3f}s")
    assert elapsed < 20.0