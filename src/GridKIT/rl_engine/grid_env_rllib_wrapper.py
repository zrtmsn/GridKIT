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
    ChargingAction,
    EpisodeMetrics,
    SimResult
)

class GridEnvRLlibWrapper(MultiAgentEnv):
    """#TODO überarbeiten
    Multi-Agent Wrapper for GridKIT environments compatible with RLlib v2+.
    
    Architecture Principle:
    - Does NOT create the environment itself.
    - Receives an implemented GridEnvProtocol via Dependency Injection. #TODO überarbeiten
    - Translates between Core models (Observation, ChargingAction) and 
      RLlib formats (numpy arrays, dicts).
    """

    metadata = {"render_modes": []}

    def __init__(self, env: GridEnvProtocol, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the wrapper.
        Args:
            env: A concrete implementation of GridEnvProtocol
            config: Optional RLlib config dict (unused for initialization logic here).
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

        # Define Spaces for RLlib (Fixed based on our Core constants)
        # Observation: 5 floats (soc_progress, time_urgency, price, base_load, temp)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        # Action: Discrete 3 (OFF, HALF, FULL)
        self.action_space = gym.spaces.Discrete(3)

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
        Translates Core Observations to numpy arrays for RLlib.
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

        # Convert Core Observation objects to numpy arrays
        np_obs_dict = {
            agent_id: self._obs_to_numpy(obs) 
            for agent_id, obs in obs_dict.items()
        }

        # Info dict (can contain initial metadata) #TODO überarbeiten
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
        1. Convert RLlib actions (int) to Core ChargingAction enums.
        2. Call env.step() on the injected environment.
        3. Store results for metrics.
        4. Convert outputs to RLlib format.
        """
        # 1. Convert Actions
        charging_actions: Dict[str, ChargingAction] = {}
        for agent_id, action_int in actions.items():
            # Map 0->OFF, 1->HALF, 2->FULL
            charging_actions[agent_id] = ChargingAction(action_int)

        # 2. Call Inner Environment
        # The magic happens here: We call the interface, unaware of the implementation
        step_results_dict, power_flow_result = self._env.step(charging_actions)

        # 3. Store for Metrics/Logging
        self._last_step_results = step_results_dict
        self._last_power_flow = power_flow_result
        self._current_step += 1

        # 4. Prepare Return Values for RLlib
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
        Helper: Convert Core Observation model to normalized numpy array.
        Ensures values are clamped to [0, 1] where appropriate for NN stability.
        """
        raw_values = obs.to_array()
        
        # Normalize specific fields if not already pre-processed
        # Index 0: soc_progress
        # Index 3: base_load_kw 
        # Index 4: temp
        
        # Converstion relies on core's to_array() structure
        return np.array(raw_values, dtype=np.float32)

    def get_latest_metrics(self) -> Tuple[Dict[str, StepResult], Optional[PowerFlowResult]]:
        """
        Helper method to retrieve the last step's detailed results.
        Useful for external loggers or the Dashboard to construct SimResult/EpisodeMetrics.
        """
        return self._last_step_results, self._last_power_flow

    # --- MultiAgentEnv Required Properties ---
    @property
    def agents(self) -> List[str]:
        return self._agents if hasattr(self, '_agents') else self._agent_ids
    
    @agents.setter
    def agents(self, value: List[str]):
        self._agents = value

    @property
    def possible_agents(self) -> List[str]:
        return self._agent_ids
    
    @possible_agents.setter
    def possible_agents(self, value: List[str]):
        pass # Usually static

    @property
    def observation_spaces(self) -> Dict[str, gym.spaces.Space]:
        return {agent_id: self.observation_space for agent_id in self._agent_ids}

    @property
    def action_spaces(self) -> Dict[str, gym.spaces.Space]:
        return {agent_id: self.action_space for agent_id in self._agent_ids}