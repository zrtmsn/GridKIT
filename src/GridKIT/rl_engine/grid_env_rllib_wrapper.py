# rl_engine/grid_env_rllib_wrapper.py
"""
GridEnv Wrapper for RLlib.
"""


from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from ray.rllib.env import MultiAgentEnv
import json          
import tempfile     
from pathlib import Path  
import ray           

# Import only from core

from GridKIT.core.protocols import GridEnvProtocol
from GridKIT.core.models import (
    Observation,
    StepResult,
    PowerFlowResult,
    ChargingAction,
)

class GridEnvRLlibWrapper(MultiAgentEnv):
    """
    Multi-Agent Wrapper for GridKIT environments compatible with RLlib v2+.
    
    How it works:
    - You hand over an existing Environment object when creating this wrapper.
    - The wrapper translates data formats: 
      Core Models (Observation, ChargingAction) <-> RLlib Formats (numpy arrays, dicts).
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
        
        # Assign worker ID using Ray's internal worker ID (unique across all workers)
        try:
            worker_hex = ray.worker.global_worker.worker_id.hex()[:4]
            self._worker_id = f"w{worker_hex}"
        except Exception:
            # Fallback if Ray worker ID not available
            import random
            self._worker_id = f"w{random.randint(1000, 9999)}"
        
        # Episode counter for this worker instance
        self._episode_counter: int = 0

        # Buffers to store latest results for metrics extraction later
        self._last_step_results: Dict[str, StepResult] = {}
        self._last_power_flow: Optional[PowerFlowResult] = None
        self._current_step: int = 0

   # File-based logging setup
        self._log_dir = Path(tempfile.gettempdir()) / "gridkit_rl_logs" #TODO change this path to a path in the Repo
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._episode_id: Optional[str] = None
        self._step_log_file: Optional[Path] = None
        self._step_data_buffer: List[Dict[str, Any]] = []

        # Define observation and action spaces using constants from core
        from GridKIT.core.config import settings
        from GridKIT.core import constants as const 
        #TODO remark that OBS / ACTION DIM need to be accessible from core/settings, otherwise constants import still required.
        
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(const.OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(const.ACTION_DIM)  # OFF=0, HALF=1, FULL=2
        #TODO add option for the user to pass a boolean when initializing the wrapper 
        # to determine wether to run in discrete or continuous mode
        #TODO add functinality for continuous mode

        # For MultiAgentEnv compatibility
        self.possible_agents = self._agent_ids
        self.agents = self._agent_ids

    def _start_new_episode(self):
        """Initialize file-based logging for a new episode."""
        self._episode_counter += 1
        self._episode_id = f"{self._worker_id}_{self._episode_counter:04d}"
        self._step_log_file = self._log_dir / f"episode_{self._episode_id}.jsonl"
        self._step_data_buffer = []
        
        # Write metadata as first line (Front-Matter) - always overwrite to start fresh
        with open(self._step_log_file, 'w') as f:
            f.write(json.dumps({
                '__metadata__': True,
                'worker_id': self._worker_id,
                'episode_num': self._episode_counter,
            }) + '\n')

    def _write_step_to_file(self, step_data: Dict[str, Any]):
        """Append step data to the current episode's log file."""
        if self._step_log_file:
            with open(self._step_log_file, 'a') as f:
                f.write(json.dumps(step_data) + '\n')
        
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

        # Start new episode logging
        self._start_new_episode()

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
        1. Convert RLlib actions (int) to ChargingAction enums.
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
        step_results_dict, power_flow_result = self._env.step(charging_actions)

        # 3. Store for Metrics/Logging
        self._last_step_results = step_results_dict
        self._last_power_flow = power_flow_result

        # Get the current step from the environment
        env_current_step = self._env._current_step
        # Log PowerFlowResult ONCE per timestep (before agent steps)
        if self._last_power_flow:
            self._write_step_to_file({
                '__powerflow__': True,  # Marker to identify power flow entries
                'timestep': env_current_step,
                'transformer_loading_pu': self._last_power_flow.transformer_loading_pu,
                'line_loadings_pu': self._last_power_flow.line_loadings_pu,
                'bus_voltages_pu': self._last_power_flow.bus_voltages_pu,
                'curtailment_applied': self._last_power_flow.curtailment_applied,
                'curtailed_power_kw': self._last_power_flow.curtailed_power_kw,
            })

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

            # Buffer step data for file writing (only real agents, not __all__)
            self._step_data_buffer.append({
                'step': env_current_step,
                'agent_id': agent_id,
                'reward': result.reward,
                'done': result.done,
                'truncated': result.truncated,
                'observation': result.observation.to_array() if result.observation else None,
            })
            
            # Write step to file (only real agents, not __all__)
            # Use 0-indexed step count (same as environment.py) for consistency
            # NOTE: transformer_loading_pu is logged separately via __powerflow__ entry
            self._write_step_to_file({
                'step': env_current_step, 
                'agent_id': agent_id, 
                'reward': result.reward, 
                'done': result.done,
                'curtailed_kw': result.info.get('curtailed_kw', 0.0),
                'soc_progress': result.observation.soc_progress,  # For SoC satisfaction calculation
            })

        # Global flags required by RLlib v2.x
        terminateds["__all__"] = episode_done
        truncateds["__all__"] = False
        infos["__all__"] = {}

        return observations, rewards, terminateds, truncateds, infos

    def _obs_to_numpy(self, obs: Observation) -> np.ndarray:
        """
        Helper: Convert Observation model to normalized numpy array.
        Ensures values are clamped to [0, 1] where appropriate for NN stability.
        """
        raw_values = obs.to_array()
        
        # Normalize specific fields if not already pre-processed
        # Index 0: soc_progress
        # Index 3: base_load_kw 
        # Index 4: temp
        
        # This conversion relies on the consistent structure of Observation.to_array().
        return np.array(raw_values, dtype=np.float32)

    def get_latest_metrics(self) -> Tuple[Dict[str, StepResult], Optional[PowerFlowResult]]:
        """
        Helper method to retrieve the last step's detailed results.
        Useful for external loggers or the Dashboard to construct SimResult/EpisodeMetrics.
        """
        return self._last_step_results, self._last_power_flow

    
    @classmethod
    def clear_logs(cls):
        """
        Remove ALL log files from the gridkit_rl_logs directory.
        """
        log_dir = Path(tempfile.gettempdir()) / "gridkit_rl_logs"
        if log_dir.exists():
            # Delete all files in the log directory
            for f in log_dir.iterdir():
                if f.is_file():
                    f.unlink()

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
        """Returns a dict mapping each agent ID to its observation space."""
        return {agent_id: self.observation_space for agent_id in self._agent_ids}

    @property
    def action_spaces(self) -> Dict[str, gym.spaces.Space]:
        """Returns a dict mapping each agent ID to its action space."""
        return {agent_id: self.action_space for agent_id in self._agent_ids}
