# grid_model/test_environment.py
import pytest
from grid_model.environment import GridEnv
from core.models import GridNetwork, Observation, EVState
import core.constants as const


@pytest.fixture
def env():
    return GridEnv()


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

