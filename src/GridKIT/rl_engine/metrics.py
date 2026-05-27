"""
Metrics Collection for GridKIT Training.

This module provides tools to aggregate raw step data into meaningful
episode-level metrics (EpisodeMetrics) and full simulation records (SimResult).
"""

from typing import List, Dict, Optional
import numpy as np

from GridKIT.core.models import (
    StepResult,
    PowerFlowResult,
    EpisodeMetrics,
    SimResult,
    EVState
)

class EpisodeLogger:
    """
    Collects data for a single training episode and computes final metrics.
    
    Usage:
        logger = EpisodeLogger()
        # Inside loop: logger.add_step(step_results, power_flow)
        # At end: metrics, sim_result = logger.finalize(episode_index)
    """
    
    def __init__(self):
        self.step_results_history: List[Dict[str, StepResult]] = []
        self.power_flows_history: List[PowerFlowResult] = []
        self.total_reward = 0.0
        self.curtailment_events = 0
        self.agents_reached_target = 0
        self.total_agents = 0
        self.max_transformer_load = 0.0

    def add_step(self, step_results: Dict[str, StepResult], power_flow: PowerFlowResult):
        """Record data from a single timestep."""
        self.step_results_history.append(step_results)
        self.power_flows_history.append(power_flow)
        
        # Aggregate immediate metrics
        for result in step_results.values():
            self.total_reward += result.reward
            
        # Count curtailment events (if any agent was curtailed)
        if any(info.get("curtailed_kw", 0) > 0 for r in step_results.values() for info in [r.info]):
            self.curtailment_events += 1
            
        # Track peak transformer load
        if power_flow.transformer_loading_pu > self.max_transformer_load:
            self.max_transformer_load = power_flow.transformer_loading_pu

    def finalize(self, episode_index: int) -> tuple[EpisodeMetrics, SimResult]:
        """
        Compute final metrics and build the SimResult object.
        Call this at the end of an episode.
        """
        # Calculate SoC satisfaction rate from the last step results
        if self.step_results_history:
            last_step = self.step_results_history[-1]
            total_agents = len(last_step)
            reached_target = 0
            
            # Note: This is a simplified check. In a real scenario, you might need 
            # to check the EVState inside the environment or pass departure info.
            # For now, we assume 'done' implies checking target status if logic exists.
            # A more robust way: Check if reward includes the completion bonus.
            # Here we just count agents that are 'done' as a placeholder until 
            # we have explicit SoC tracking in StepResult.info if needed.
            # Better: Let's assume we can derive it from the info or reward structure.
            # For this stub, we set a dummy value or calculate based on available data.
            # TODO: Refine this logic based on exact GridEnv termination criteria.
            
            # Hack for now: Count agents where reward > threshold (indicating bonus)? 
            # Or simply return 0.0 until Env passes explicit SoC status in info.
            # Let's assume for now we don't have perfect info here without Env state access.
            # We will return 0.0 or implement a callback in Env later.
            soc_satisfaction_rate = 0.0 # Placeholder
            
            self.total_agents = total_agents
        else:
            soc_satisfaction_rate = 0.0
            self.total_agents = 0

        # Build EpisodeMetrics
        metrics = EpisodeMetrics(
            episode=episode_index,
            mean_episode_reward=self.total_reward / max(1, self.total_agents), # Avg per agent approx
            soc_satisfaction_rate=soc_satisfaction_rate,
            curtailment_events=self.curtailment_events,
            transformer_peak_loading_pu=self.max_transformer_load,
            epsilon=None # IPPO doesn't use epsilon, only DQN
        )
        
        # Build SimResult
        # SimResult expects a list of PowerFlowResults and final SoC per agent
        # We construct a dummy final_soc dict for now as Env doesn't expose it directly in StepResult
        final_soc_per_agent = {} # Would need extraction from Env state
        
        sim_result = SimResult(
            episode_metrics=metrics,
            timestep_results=self.power_flows_history,
            final_soc_per_agent=final_soc_per_agent
        )
        
        return metrics, sim_result