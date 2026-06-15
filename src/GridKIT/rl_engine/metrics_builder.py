"""
Metrics Builder for GridKIT Training.

Metrics Builder: Post-training metrics construction from RL training logs.

This module provides functions and a service class to build EpisodeMetrics,
SimResult, and IterationMetrics objects from JSONL log files created during
RLlib training.

Naming convention:
- "build_*" = Constructs Python objects for RAM usage (after training completes)
- LocalTrainer._save_raw_iteration_results() writes raw RLlib dicts directly (no objects)
- MetricsBuilder.build_iteration_metrics() loads raw dicts and builds objects

MetricsBuilder Service Class serves as the main entry point.

"""


import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from GridKIT.core.models import EpisodeMetrics, SimResult, PowerFlowResult, IterationMetrics, AllMetrics
from dataclasses import dataclass


# =============================================================================
# Build IterationMetrics from RLlib result (helper function)
# =============================================================================

def build_iteration_metrics_from_rl_result(
    result: Dict[str, Any], 
    iteration: int
) -> Optional[IterationMetrics]:
    """
    Build IterationMetrics from RLlib result dict.
    
    This is a helper function used by MetricsBuilder.build_iteration_metrics()
    to construct IterationMetrics objects from raw RLlib result dicts.
    
    Args:
        result: RLlib result dict from algo.train()
        iteration: Current iteration number
        
    Returns:
        IterationMetrics object or None if required data missing
    """
    if 'learners' not in result:
        return None
    
    policy_data = result.get('learners', {}).get('household_policy', {})
    env_runners = result.get('env_runners', {})
    fault_tolerance = result.get('fault_tolerance', {})
    
    # Extract per-agent metrics (IPPO multi-agent)
    agent_entropies: Optional[Dict[str, float]] = None
    agent_policy_losses: Optional[Dict[str, float]] = None
    
    for key, value in policy_data.items():
        if key.startswith('agent_') and '/entropy' in key:
            if agent_entropies is None:
                agent_entropies = {}
            agent_id = key.split('/')[0]
            agent_entropies[agent_id] = float(value)
        elif key.startswith('agent_') and '/policy_loss' in key:
            if agent_policy_losses is None:
                agent_policy_losses = {}
            agent_id = key.split('/')[0]
            agent_policy_losses[agent_id] = float(value)
    
    return IterationMetrics(
        iteration=iteration,
        policy_loss=policy_data.get('policy_loss', 0.0),
        entropy=policy_data.get('entropy', 0.0),
        mean_kl=policy_data.get('mean_kl', 0.0),
        vf_loss=policy_data.get('vf_loss', 0.0),
        vf_explained_var=policy_data.get('vf_explained_var', 0.0),
        cur_lr=policy_data.get('default_optimizer_learning_rate', 0.0),
        grad_norm=fault_tolerance.get('grad_norm', None),
        timesteps_this_iter=result.get('num_env_steps_sampled_lifetime', 0),
        episodes_this_iter=env_runners.get('episodes_this_iter', 0),
        total_steps=result.get('num_env_steps_sampled_lifetime', 0),
        episode_reward_mean=env_runners.get('episode_return_mean', 0.0),
        episode_reward_min=env_runners.get('episode_return_min', 0.0),
        episode_reward_max=env_runners.get('episode_return_max', 0.0),
        episode_len_mean=env_runners.get('episode_len_mean', 0.0),
        agent_entropies=agent_entropies,
        agent_policy_losses=agent_policy_losses,
        curr_entropy_coeff=policy_data.get('curr_entropy_coeff'),
    )


# =============================================================================
# Build EpisodeMetrics from JSONL file
# =============================================================================

def build_episode_metrics_from_file(log_file: Path) -> EpisodeMetrics:
    """
    Reads a single episode log file and builds an EpisodeMetrics object.
    
    Used by: EpisodeMetricsCollector.collect_all()
    """
    steps = []
    powerflow_entries = []
    metadata = {}
    
    with open(log_file) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('__metadata__'):
                    metadata = data
                    continue
                if data.get('__powerflow__'):
                    powerflow_entries.append(data)
                    continue
                steps.append(data)
            except json.JSONDecodeError:
                continue
    
    if not steps:
        return EpisodeMetrics(
            episode=0,
            mean_episode_reward=0.0,
            soc_satisfaction_rate=0.0,
            curtailment_events=0,
            transformer_peak_loading_pu=0.0,
        )
    
    episode_num = metadata.get('episode_num', 0)
    if episode_num == 0:
        try:
            episode_num = int(log_file.stem.split('_')[2])
        except (IndexError, ValueError):
            episode_num = 0
    
    total_reward = sum(s.get('reward', 0) for s in steps)
    curtail_count = sum(1 for pf in powerflow_entries if pf.get('curtailment_applied', False))
    
    def safe_trafo(pf: dict) -> float:
        val = pf.get('transformer_loading_pu', 0.0)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 0.0
        return val
    
    transformer_peak = max((safe_trafo(pf) for pf in powerflow_entries), default=0.0)
    
    final_soc_progress_per_agent: Dict[str, float] = {}  # type: ignore
    for s in steps:
        agent_id = s.get('agent_id')
        if agent_id and agent_id != '__all__' and s.get('done', False):
            soc_progress = s.get('soc_progress', 0.0)
            final_soc_progress_per_agent[agent_id] = soc_progress
    
    satisfied_count = sum(1 for soc_p in final_soc_progress_per_agent.values() if soc_p >= 1.0)
    soc_satisfaction_rate = satisfied_count / len(final_soc_progress_per_agent) if final_soc_progress_per_agent else 0.0
    
    return EpisodeMetrics(
        episode=episode_num,
        mean_episode_reward=total_reward,
        soc_satisfaction_rate=soc_satisfaction_rate,
        curtailment_events=curtail_count,
        transformer_peak_loading_pu=transformer_peak,
        epsilon=None,
        entropy=None,
    )


# =============================================================================
# Build SimResult from JSONL file
# =============================================================================

def build_sim_result_from_file(
    log_file: Path,
    network_id: str = "stub_network",
    ev_penetration: float = 1.0,
) -> SimResult:
    """
    Reads a single episode log file and builds a SimResult object.
    
    Used by: EpisodeMetricsCollector.get_sim_results_for_worker()
    """
    steps = []
    powerflow_entries = []
    metadata = {}
    
    with open(log_file) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('__metadata__'):
                    metadata = data
                    continue
                if data.get('__powerflow__'):
                    powerflow_entries.append(data)
                    continue
                steps.append(data)
            except json.JSONDecodeError:
                continue
    
    episode_num = metadata.get('episode_num', 0)
    if episode_num == 0:
        try:
            episode_num = int(log_file.stem.split('_')[2])
        except (IndexError, ValueError):
            episode_num = 0
    
    timestep_results: List[PowerFlowResult] = []
    for pf in powerflow_entries:
        t = pf.get('timestep', 0)
        trafo_loading = pf.get('transformer_loading_pu', 0.0)
        if trafo_loading is None or (isinstance(trafo_loading, float) and math.isnan(trafo_loading)):
            trafo_loading = 0.0
        
        pf_result = PowerFlowResult(
            timestep=t,
            transformer_loading_pu=trafo_loading,
            line_loadings_pu=pf.get('line_loadings_pu', {}),
            bus_voltages_pu=pf.get('bus_voltages_pu', {}),
            curtailment_applied=pf.get('curtailment_applied', False),
            curtailed_power_kw=pf.get('curtailed_power_kw', {}),
        )
        timestep_results.append(pf_result)
    
    timestep_results.sort(key=lambda x: x.timestep)
    
    final_soc_per_agent: Dict[str, float] = {}
    for s in steps:
        agent_id = s.get('agent_id')
        if agent_id and agent_id != '__all__' and s.get('done', False):
            soc_progress = s.get('soc_progress', 0.0)
            final_soc_per_agent[agent_id] = soc_progress * 0.8
    
    metrics = build_episode_metrics_from_file(log_file)
    
    return SimResult(
        episode=episode_num,
        network_id=network_id,
        ev_penetration=ev_penetration,
        timestep_results=timestep_results,
        final_soc_per_agent=final_soc_per_agent,
        metrics=metrics,
    )


# =============================================================================
# EpisodeMetricsCollector (used by MetricsBuilder.build_all())
# =============================================================================

class EpisodeMetricsCollector:
    """
    Collects EpisodeMetrics and SimResult objects from RLlib training logs.
    
    Used by: MetricsBuilder._get_collector() → build_all()
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.metrics_by_worker: Dict[str, List[EpisodeMetrics]] = defaultdict(list)
        self.all_files: List[Path] = []
        self._collected = False
    
    def collect_all(self) -> None:
        """Scan log directory and collect EpisodeMetrics from all files."""
        if not self.log_dir.exists():
            return
        
        self.all_files = sorted(self.log_dir.glob("episode_w*_*.jsonl"))
        
        files_by_worker: Dict[str, List[Path]] = defaultdict(list)
        for f in self.all_files:
            try:
                parts = f.stem.split('_')
                if len(parts) >= 2:
                    worker_id = parts[1]
                    files_by_worker[worker_id].append(f)
            except (IndexError, ValueError):
                continue
        
        for worker_id, files in sorted(files_by_worker.items()):
            files_sorted = sorted(
                files, 
                key=lambda f: int(f.stem.split('_')[2]) if len(f.stem.split('_')) >= 3 else 0
            )
            
            for f in files_sorted:
                try:
                    ep_num = int(f.stem.split('_')[2])
                    if ep_num == 1:  # Skip warm-up episode
                        continue
                    
                    metrics = build_episode_metrics_from_file(f)
                    adjusted_metrics = EpisodeMetrics(
                        #adjusts episode number according to skipping the warm-up episode
                        episode=metrics.episode - 1,
                        mean_episode_reward=metrics.mean_episode_reward,
                        soc_satisfaction_rate=metrics.soc_satisfaction_rate,
                        curtailment_events=metrics.curtailment_events,
                        transformer_peak_loading_pu=metrics.transformer_peak_loading_pu,
                        epsilon=metrics.epsilon,
                        entropy=metrics.entropy,
                    )
                    self.metrics_by_worker[worker_id].append(adjusted_metrics)
                except (IndexError, ValueError):
                    continue
        
        self._collected = True
    
    def get_all_metrics_skip_first(self) -> List[EpisodeMetrics]:
        """Get all EpisodeMetrics from all workers, sorted by episode number."""
        if not self._collected:
            self.collect_all()
        
        all_metrics: List[EpisodeMetrics] = []
        for metrics_list in self.metrics_by_worker.values():
            all_metrics.extend(metrics_list)
        
        return sorted(all_metrics, key=lambda m: m.episode)
    
    def get_sim_results_for_worker(self, worker_id: str) -> List[SimResult]:
        """Build and return SimResult objects for a specific worker."""
        if not self._collected:
            self.collect_all()
        
        if worker_id not in self.metrics_by_worker:
            return []
        
        metrics_list = self.metrics_by_worker[worker_id]
        
        worker_files = sorted(
            [f for f in self.all_files if f.stem.split('_')[1] == worker_id],
            key=lambda f: int(f.stem.split('_')[2]) if len(f.stem.split('_')) >= 3 else 0
        )
        
        worker_files_filtered = [
            f for f in worker_files
            if len(f.stem.split('_')) >= 3 and int(f.stem.split('_')[2]) > 1
        ]
        
        sim_results = []
        for i, metrics in enumerate(metrics_list):
            if i >= len(worker_files_filtered):
                break
            
            log_file = worker_files_filtered[i]
            sim_result = build_sim_result_from_file(
                log_file,
                network_id="stub_network",
                ev_penetration=1.0,
            )
            sim_result.episode = metrics.episode
            sim_result.metrics.episode = metrics.episode
            sim_results.append(sim_result)
        
        return sim_results

# =============================================================================
# MetricsBuilder Service Class (main entry point)
# =============================================================================

class MetricsBuilder:
    """
    Service class for building metrics objects from RL training logs.
    
    Used by: rl_control.py → run_training()
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self._collector: Optional[EpisodeMetricsCollector] = None
    
    def _get_collector(self) -> EpisodeMetricsCollector:
        """Lazy-load the collector."""
        if self._collector is None:
            self._collector = EpisodeMetricsCollector(self.log_dir)
            self._collector.collect_all()
        return self._collector
    
    def build_iteration_metrics(self) -> List[IterationMetrics]:
        """
        Build IterationMetrics objects from raw RLlib result dicts stored in JSON file.
        
        Loads raw data from iteration_metrics_raw.json (written by LocalTrainer)
        and builds IterationMetrics objects for RAM usage.
        
        Returns:
            List of IterationMetrics objects, one per training iteration
        """
        output_file = self.log_dir / "iteration_metrics_raw.json"
        
        if not output_file.exists():
            return []
        
        with open(output_file, 'r') as f:
            raw_results = json.load(f)
        
        # Build IterationMetrics objects from raw RLlib result dicts
        return [
            build_iteration_metrics_from_rl_result(raw, i + 1)
            for i, raw in enumerate(raw_results)
        ]
    
    def build_all(self) -> AllMetrics:
        """
        Build all metrics at once.
        
        This is the MAIN ENTRY POINT used by rl_control.py!
        """
        collector = self._get_collector()
        
        episode_metrics = collector.get_all_metrics_skip_first()
        
        sim_results: List[SimResult] = []
        for worker_id in collector.metrics_by_worker.keys():
            sim_results.extend(collector.get_sim_results_for_worker(worker_id))
        
        iteration_metrics = self.build_iteration_metrics()
        
        return AllMetrics(
            episode_metrics=episode_metrics,
            sim_results=sim_results,
            iteration_metrics=iteration_metrics,
        )