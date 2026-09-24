# rl_engine/grid_env_rllib_wrapper.py
"""
GridEnv Wrapper for RLlib.
"""


from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from ray.rllib.env import MultiAgentEnv

# Import only from core

from GridKIT.core.protocols import GridEnvProtocol
from GridKIT.core.models import (
    Observation,
    StepResult,
    PowerFlowResult,
    device_of,
)
from GridKIT.rl_engine.obs_norm import normalize_observation_multidevice

class GridEnvRLlibWrapper(MultiAgentEnv):
    """
    Multi-Agent Wrapper for GridKIT environments compatible with RLlib v2+.
    
    How it works:
    - You hand over an existing Environment object when creating this wrapper.
    - The wrapper translates data formats:
      Core Models (Observation, per-device action indices) <-> RLlib Formats (numpy arrays, dicts).
    - It handles the communication loop without knowing the internal logic of the Environment.
    """

    metadata = {"render_modes": []}

    def __init__(self, env: GridEnvProtocol, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the wrapper.
        
        Args:
            env: An already created Environment object (must implement GridEnvProtocol).
                 We simply store it as reference and use it for all steps.
            config: Optional RLlib config dict (not used for setup here).
        """
        super().__init__()
        
        # Store the injected environment
        self._env = env
        
        # Extract agent IDs from the environment once
        self._agent_ids: List[str] = self._env.agent_ids
        
        # Buffers to store latest results for metrics extraction later
        self._last_step_results: Dict[str, StepResult] = {}
        self._last_power_flow: Optional[PowerFlowResult] = None
        self._current_step: int = 0

        # Define observation and action spaces using constants from core.
        #
        # Agents are per-device (ev / battery / hp), and device types do NOT all
        # share the same action space: EV and battery have 3 discrete actions,
        # heat pump only has 2 (OFF/HEAT — see constants.ACTION_DIM_BY_DEVICE). A
        # single shared Discrete(ACTION_DIM) space would let RLlib sample an
        # out-of-range action index for an HP agent and crash inside GridEnv.step().
        # Observation width IS uniform across device types (soc_progress/
        # time_urgency carry each agent's own device state) — we use the 10-dim
        # multi-device vector (OBS_DIM_MULTIDEVICE) via obs_norm, properly
        # normalized to [0,1] (unlike the legacy 7-dim to_array(), whose raw
        # values — e.g. temperature -10..40, price ~0.01..0.6 — violate the
        # [0,1] Box this space declares) — so only the action space needs to be
        # per-device.
        from GridKIT.core import constants as const

        self._obs_space_per_agent = gym.spaces.Box(
            low=0.0, high=1.0, shape=(const.OBS_DIM_MULTIDEVICE,), dtype=np.float32
        )
        self._action_space_by_device = {
            device: gym.spaces.Discrete(dim)
            for device, dim in const.ACTION_DIM_BY_DEVICE.items()
        }

        # RLlib multi-agent contract for heterogeneous per-agent spaces: expose
        # observation_space/action_space as agent_id-keyed Dict spaces and flag
        # that they're already in that "preferred" per-agent format.
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict(
            {aid: self._obs_space_per_agent for aid in self._agent_ids}
        )
        self.action_space = gym.spaces.Dict(
            {aid: self._action_space_by_device[device_of(aid)] for aid in self._agent_ids}
        )
        # Note: the wrapper currently runs in discrete mode only; a continuous
        # mode is not implemented yet. Left for future work.


        # For MultiAgentEnv compatibility
        self.possible_agents = self._agent_ids
        self.agents = self._agent_ids

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment and return initial observations.
        Translates Observations to numpy arrays for RLlib.
        """
        if seed is not None:
            # Pass seed to inner env if supported, otherwise ignore
            try:
                obs_dict = self._env.reset(seed=seed)
            except TypeError:
                obs_dict = self._env.reset()
        else:
            obs_dict = self._env.reset()

        self._current_step = 0
        self._last_step_results = {}
        self._last_power_flow = None

        # Convert Observation objects to numpy arrays
        np_obs_dict = {
            agent_id: self._obs_to_numpy(obs) 
            for agent_id, obs in obs_dict.items()
        }

        # Info dict (required by RLlib MultiAgentEnv interface)
        info_dict = {agent_id: {} for agent_id in self._agent_ids}
        info_dict["__all__"] = {}

        return np_obs_dict, info_dict

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray], 
        Dict[str, float], 
        Dict[str, bool], 
        Dict[str, bool], 
        Dict[str, Any]
    ]:
        """
        Execute one step in the environment.
        1. Call env.step() on the injected environment (GridEnv takes plain
           per-device action indices directly — decoding into ChargingAction/
           BatteryAction/HPAction happens inside GridEnv itself, since which
           enum applies depends on each agent's device type).
        2. Store results for metrics.
        3. Convert outputs to RLlib format.
        """
        # 1. Call Inner Environment
        step_results_dict, power_flow_result = self._env.step(actions)

        # 2. Store for Metrics/Logging
        self._last_step_results = step_results_dict
        self._last_power_flow = power_flow_result
        self._current_step += 1

        # 3. Prepare Return Values for RLlib
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        episode_done = False # Global flag

        for agent_id, result in step_results_dict.items():
            # Observation
            observations[agent_id] = self._obs_to_numpy(result.observation)
            
            # Reward
            rewards[agent_id] = result.reward
            
            # Done flags
            # 'terminated' means natural end (goal reached or time up)
            terminateds[agent_id] = result.done
            # 'truncated' usually for time limits forced by external factors
            truncateds[agent_id] = result.truncated
            
            if result.done:
                episode_done = True

            # Info: Pass through extra data (like curtailment info)
            # We also stash the full objects here temporarily if needed for callbacks
            infos[agent_id] = result.info

        # Global flags required by RLlib v2.x
        terminateds["__all__"] = episode_done
        truncateds["__all__"] = False
        infos["__all__"] = {}

        return observations, rewards, terminateds, truncateds, infos

    def _obs_to_numpy(self, obs: Observation) -> np.ndarray:
        """
        Helper: Convert Observation model to a normalized [0,1] numpy array.
        Delegates to obs_norm.normalize_observation_multidevice — the single
        source of truth shared with RLlibPolicyAdapter, so train-time and
        eval-time inputs match.
        """
        return normalize_observation_multidevice(obs)

    def get_latest_metrics(self) -> Tuple[Dict[str, StepResult], Optional[PowerFlowResult]]:
        """
        Helper method to retrieve the last step's detailed results.
        Useful for external loggers or the Dashboard to construct SimResult/EpisodeMetrics.
        """
        return self._last_step_results, self._last_power_flow

    # ----------------------------------------------------------------------
    # RLlib v2.x Interface Requirements
    # ----------------------------------------------------------------------
    # RLlib strictly expects these specific properties to exist and be queryable.
    # We implement them as @properties (instead of simple attributes) to:
    # 1. Maintain control over internal state (e.g., distinguishing between 
    #    'possible_agents' vs. active 'agents').
    # 2. Safely intercept external assignments (setters) if RLlib tries to 
    #    modify the agent list dynamically during execution.
    # 3. Ensure dynamic calculation if needed, rather than static storage.
    # ----------------------------------------------------------------------

    @property
    def agents(self) -> List[str]:
        """Returns the list of currently active agents."""
        # Fallback to all possible agents if no specific active list is set
        return self._agents if hasattr(self, '_agents') else self._agent_ids
    
    @agents.setter
    def agents(self, value: List[str]):
        """Allows RLlib to update the active agent list safely."""
        self._agents = value
  
    @property
    def possible_agents(self) -> List[str]:
        """Returns the static list of all potential agent IDs in this environment."""
        return self._agent_ids
    
    @possible_agents.setter
    def possible_agents(self, value: List[str]):
        """Placeholder setter to prevent errors if RLlib attempts assignment."""
        pass # Usually static, so we ignore external writes
  
    @property
    def observation_spaces(self) -> Dict[str, gym.spaces.Space]:
        """Returns a dict mapping each agent ID to its observation space.

        Uniform across device types (Observation.to_array() is always OBS_DIM
        wide) — unlike action_spaces below, which does vary by device.
        """
        return {agent_id: self._obs_space_per_agent for agent_id in self._agent_ids}

    @property
    def action_spaces(self) -> Dict[str, gym.spaces.Space]:
        """Returns a dict mapping each agent ID to its (device-type-specific) action space."""
        return {agent_id: self._action_space_by_device[device_of(agent_id)] for agent_id in self._agent_ids}
