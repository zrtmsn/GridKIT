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

@pytest.fixture
def all_half(env):
    return {agent_id: ChargingAction.HALF for agent_id in env.agent_ids}
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
        
# trivially passes until random EV sampling is implemented
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
# §14a curtailment is triggered when the pre-charging power flow puts the
# transformer or any line above its overload threshold (core.constants).
# The stub network (5 households, 400 kW trafo, ~97 kW lines) never overloads
# under normal FULL charging, so tests that need to force curtailment
# temporarily lower the thresholds via monkeypatch rather than editing
# core/constants.py permanently.

def test_no_curtailment_when_not_overloaded(env, all_off):
    _, power_flow = env.step(all_off)
    assert not power_flow.curtailment_applied

def test_no_curtailment_at_full_charge_under_default_thresholds(env, all_full):
    # 5 households at FULL (7.4 kW each) stay well within the stub network's
    # capacity, so §14a should not trigger under the real-world thresholds
    _, power_flow = env.step(all_full)
    assert not power_flow.curtailment_applied
    assert power_flow.curtailed_power_kw == {}

def test_curtailment_triggers_on_transformer_overload(env, all_full, monkeypatch):
    monkeypatch.setattr(const, "TRANSFORMER_OVERLOAD_THRESHOLD", 0.0)
    _, power_flow = env.step(all_full)
    assert power_flow.curtailment_applied

def test_curtailment_triggers_on_line_overload(env, all_full, monkeypatch):
    monkeypatch.setattr(const, "LINE_OVERLOAD_THRESHOLD", 0.0)
    _, power_flow = env.step(all_full)
    assert power_flow.curtailment_applied

def test_curtailment_caps_ev_power_at_full_charge(env, all_full, monkeypatch):
    monkeypatch.setattr(const, "TRANSFORMER_OVERLOAD_THRESHOLD", 0.0)
    step_results, power_flow = env.step(all_full)
    expected_curtailed = const.EV_POWER_FULL_KW - const.MAX_CONTROLLED_POWER_KW
    for agent_id in env.agent_ids:
        assert power_flow.curtailed_power_kw[agent_id] == pytest.approx(expected_curtailed)
        assert step_results[agent_id].info["curtailed_kw"] == pytest.approx(expected_curtailed)

def test_curtailment_does_not_cap_power_already_below_limit(env, all_half, monkeypatch):
    # HALF (3.7 kW) is already below MAX_CONTROLLED_POWER_KW (4.2 kW), so a
    # forced overload should leave it untouched even though the flag is set
    monkeypatch.setattr(const, "TRANSFORMER_OVERLOAD_THRESHOLD", 0.0)
    step_results, power_flow = env.step(all_half)
    assert power_flow.curtailment_applied
    assert power_flow.curtailed_power_kw == {}
    for agent_id in env.agent_ids:
        assert step_results[agent_id].info["curtailed_kw"] == 0.0

def test_curtailment_flagged_but_nothing_to_cut_when_off(env, all_off, monkeypatch):
    # base load alone trips a zero threshold, but there is no EV power to curtail
    monkeypatch.setattr(const, "TRANSFORMER_OVERLOAD_THRESHOLD", 0.0)
    step_results, power_flow = env.step(all_off)
    assert power_flow.curtailment_applied
    assert power_flow.curtailed_power_kw == {}
    for agent_id in env.agent_ids:
        assert step_results[agent_id].info["curtailed_kw"] == 0.0

def test_curtailment_reduces_soc_gain_vs_uncurtailed(env, all_full, monkeypatch):
    soc_before = {a: env._evs[a].soc for a in env.agent_ids}
    monkeypatch.setattr(const, "TRANSFORMER_OVERLOAD_THRESHOLD", 0.0)
    env.step(all_full)

    capped_delta = const.MAX_CONTROLLED_POWER_KW * const.TIMESTEP_HOURS / const.EV_BATTERY_CAPACITY_KWH
    uncurtailed_delta = const.EV_POWER_FULL_KW * const.TIMESTEP_HOURS / const.EV_BATTERY_CAPACITY_KWH
    for agent_id in env.agent_ids:
        actual_delta = env._evs[agent_id].soc - soc_before[agent_id]
        assert actual_delta == pytest.approx(capped_delta)
        assert actual_delta < uncurtailed_delta

def test_curtailment_recomputes_power_flow_with_capped_load(env, all_full, monkeypatch):
    # the power flow returned to the caller must reflect the capped load,
    # not the overloaded load that triggered curtailment in the first place
    base_load_kw = env._h0_profile_kw.iloc[0] * env._load_multiplier
    uncapped_bus_load_mw = {
        b: (base_load_kw + const.EV_POWER_FULL_KW) / 1000.0 for b in env.agent_ids
    }
    uncapped_trafo_loading, _, _ = env._surrogate.solve(uncapped_bus_load_mw)

    monkeypatch.setattr(const, "TRANSFORMER_OVERLOAD_THRESHOLD", 0.0)
    _, power_flow = env.step(all_full)

    capped_bus_load_mw = {
        b: (base_load_kw + const.MAX_CONTROLLED_POWER_KW) / 1000.0 for b in env.agent_ids
    }
    expected_trafo_loading, expected_line_loadings, _ = env._surrogate.solve(capped_bus_load_mw)

    assert power_flow.transformer_loading_pu == pytest.approx(expected_trafo_loading)
    assert power_flow.transformer_loading_pu < uncapped_trafo_loading
    for line_id, expected in expected_line_loadings.items():
        assert power_flow.line_loadings_pu[line_id] == pytest.approx(expected)

def test_curtailment_info_matches_power_flow_result(env, all_full, monkeypatch):
    monkeypatch.setattr(const, "TRANSFORMER_OVERLOAD_THRESHOLD", 0.0)
    step_results, power_flow = env.step(all_full)
    for agent_id in env.agent_ids:
        assert step_results[agent_id].info["curtailed_kw"] == pytest.approx(
            power_flow.curtailed_power_kw.get(agent_id, 0.0)
        )

def test_curtailment_effects_report(env, all_full, monkeypatch):
    """Not a strict assertion test — prints a before/after summary of what
    curtailment changes (delivered power, SoC gain, transformer loading).
    Run with `pytest -s` to see the output."""
    base_load_kw = env._h0_profile_kw.iloc[0] * env._load_multiplier
    uncapped_bus_load_mw = {
        b: (base_load_kw + const.EV_POWER_FULL_KW) / 1000.0 for b in env.agent_ids
    }
    uncapped_trafo_loading, uncapped_lines, _ = env._surrogate.solve(uncapped_bus_load_mw)

    soc_before = {a: env._evs[a].soc for a in env.agent_ids}
    monkeypatch.setattr(const, "TRANSFORMER_OVERLOAD_THRESHOLD", 0.0)
    step_results, power_flow = env.step(all_full)

    print("\n── Curtailment effects (all agents requesting FULL charge) ──")
    print(f"transformer loading: {uncapped_trafo_loading:.3f} pu (requested) "
          f"-> {power_flow.transformer_loading_pu:.3f} pu (delivered)")
    print(f"max line loading:    {max(uncapped_lines.values()):.3f} pu (requested) "
          f"-> {max(power_flow.line_loadings_pu.values()):.3f} pu (delivered)")
    for agent_id in env.agent_ids:
        delta_soc = env._evs[agent_id].soc - soc_before[agent_id]
        print(f"{agent_id}: requested {const.EV_POWER_FULL_KW:.1f} kW, "
              f"curtailed {power_flow.curtailed_power_kw.get(agent_id, 0.0):.1f} kW, "
              f"delivered {const.EV_POWER_FULL_KW - power_flow.curtailed_power_kw.get(agent_id, 0.0):.1f} kW, "
              f"soc +{delta_soc:.4f} -> {step_results[agent_id].reward:+.4f} reward")


# ── Observations ──────────────────────────────────────────────

def test_step_observation_soc_progress_updates(env, all_full):
    step_results, _ = env.step(all_full)
    for agent_id, result in step_results.items():
        expected = env._evs[agent_id].soc / env._evs[agent_id].target_soc
        assert abs(result.observation.soc_progress - expected) < 1e-6

# ── Benchmark ──────────────────────────────────────────────

def test_step_performance(env, all_full):    
    start = time.time()
    env.step(all_full)
    elapsed = time.time() - start
    print(f"\nstep() took {elapsed:.3f}s")
    assert elapsed < 1.0