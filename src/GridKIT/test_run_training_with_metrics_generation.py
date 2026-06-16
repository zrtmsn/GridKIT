# test_run_training_metrics_generation.py
"""
Test script for generating Metrics Objects by running the training through RL_Control, the official entry point for RL training.


Short Documentation of RL_Control functionality:
1. Selects the appropriate trainer (local or HPC)
2. Runs the training loop
3. Triggers post-training metrics collection
4. Returns AllMetrics container

This testing scenario simulates an entry to RL-Engine where the Engine is already integrated into
a system that utilizes a propper UI, which is the reason why it does not print progress to console.
Instead, it generates 

    - test_get_metrics_objects_episodeMetrics.json
    - test_get_metrics_objects_simResults.json
    - test_get_metrics_objects_iterationMetrics.json

that are JSON-formats of all objects the Engine returns through the AllMetrics container after training.


Usage:
    python src/GridKIT/test_run_training_with_rl_control.py

    Check your operating system's local directory that stores temporary log files
    In Linux, that is: /tmp/gridkit_rl_logs
    #TODO migrate log file path to a relative path within the repository

"""

import sys
import os
from datetime import datetime
from pathlib import Path

# --- SETUP PATHS ---
script_dir = Path(__file__).parent  # src/GridKIT/
src_dir = script_dir.parent          # src/
sys.path.insert(0, str(src_dir))

# Changes working directory to project root for relative file access.
project_root = script_dir.parent.parent
os.chdir(project_root)

# Sets PYTHONPATH for all Ray workers.
os.environ["PYTHONPATH"] = f"{script_dir}:{src_dir}"
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"


import json
from GridKIT.grid_model.environment import GridEnv
from GridKIT.rl_engine import GridEnvRLlibWrapper, create_ippo_config
from GridKIT.core.models import AllMetrics

# Import RLControl directly - this is THE entry point!
from GridKIT.rl_engine.rl_control import RLControl, TrainerType


def main():
    # Clear old logs before training (ONLY ONCE at the very beginning!)
    GridEnvRLlibWrapper.clear_logs()
    
    # Factory: Creates a fresh Env+Wrapper for each Ray worker
    def my_env_factory(config=None):
        env = GridEnv()
        return GridEnvRLlibWrapper(env=env)

    # Create RLControl with LOCAL trainer
    control = RLControl(trainer_type=TrainerType.LOCAL)
    
    # Run training - returns AllMetrics directly!
    all_metrics = control.run_training(
        env_factory=my_env_factory,
        config_func=create_ippo_config,
        num_iterations=3,  # Short training for testing
    )
    
    # Get log directory and export metrics to JSON files
    log_dir = control.get_trainer_log_dir()
    if log_dir:
        export_metrics_to_json(all_metrics, log_dir)


def export_metrics_to_json(all_metrics: AllMetrics, log_dir: Path):
    """
    Export AllMetrics objects to separate JSON files for testing/debugging.
    
    Creates three files in the log directory:
    - test_get_metrics_objects_episodeMetrics.json
    - test_get_metrics_objects_simResults.json
    - test_get_metrics_objects_iterationMetrics.json
    
    These files are automatically cleared when clear_logs() is called.
    
    Args:
        all_metrics: AllMetrics container with all metric types
        log_dir: Directory where log files are stored
    """
    # Export EpisodeMetrics
    episode_metrics_data = [ep.model_dump() for ep in all_metrics.episode_metrics]
    episode_file = log_dir / "test_get_metrics_objects_episodeMetrics.json"
    with open(episode_file, 'w') as f:
        json.dump(episode_metrics_data, f, indent=2, default=str)
    
    # Export SimResults
    sim_results_data = [sr.model_dump() for sr in all_metrics.sim_results]
    sim_file = log_dir / "test_get_metrics_objects_simResults.json"
    with open(sim_file, 'w') as f:
        json.dump(sim_results_data, f, indent=2, default=str)
    
    # Export IterationMetrics
    iteration_metrics_data = [im.model_dump() for im in all_metrics.iteration_metrics]
    iteration_file = log_dir / "test_get_metrics_objects_iterationMetrics.json"
    with open(iteration_file, 'w') as f:
        json.dump(iteration_metrics_data, f, indent=2, default=str)


if __name__ == "__main__":
    main()