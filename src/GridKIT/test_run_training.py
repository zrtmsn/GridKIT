# test_run_training.py
"""
GridKIT Training Runner.

This script orchestrates the training process by bringing together the
three main modules:
1. grid_model: Creates the physical environment.
2. rl_engine: Provides the wrapper and configuration.
3. core: Defines the shared data models.

Usage:
    python run_training.py
"""

import sys
import os
from pathlib import Path

# --- 1. SETUP PATHS ---
# Adds src directory to Python path for imports.
# GridKIT package is located at src/GridKIT/.
script_dir = Path(__file__).parent  # src/GridKIT/
src_dir = script_dir.parent  # src/
sys.path.insert(0, str(src_dir))

# Changes working directory to project root for relative file access.
project_root = script_dir.parent.parent
os.chdir(project_root)

# Sets PYTHONPATH and working directory for all Ray workers.
# Both src/ and GridKIT/ paths are included for compatibility with different import styles.
os.environ["PYTHONPATH"] = f"{script_dir}:{src_dir}"
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"  # Prevents Ray from changing the working directory

# Imports modules cleanly.
from GridKIT.grid_model import StubNetworkBuilder
from GridKIT.grid_model.environment import GridEnv
from GridKIT.rl_engine import GridEnvRLlibWrapper
from GridKIT.rl_engine.ippo_config import create_ippo_config
from GridKIT.core import settings

# Ray & RLlib imports
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env


def main():
    print("Starting GridKIT Training Run...")

    # --- 2. INITIALIZE RAY ---
    # Initializes Ray with runtime_env to propagate PYTHONPATH to all workers.
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": f"{script_dir}:{src_dir}",
                },
                "working_dir": str(project_root),
            }
        )
    print("Ray initialized.")

    # --- 3. BUILD THE ENVIRONMENT ---
    # Creates a StubNetwork environment for Phase 1 training.
    print("Building Network Environment...")
    network_builder = StubNetworkBuilder()
    network = network_builder.build()

    # Creates the GridEnv instance (internally creates its own network).
    env = GridEnv()
    network = env.network
    print(f"   Network loaded with {len(env.agent_ids)} households.")

    # --- 4. WRAP FOR RLLIB ---
    # Injects the environment into the RLlib wrapper.
    print("Wrapping Environment for RLlib...")
    wrapper = GridEnvRLlibWrapper(env=env)

    # Registers the wrapped environment with Ray.
    env_name = "GridKit-Train-v0"

    # Defines a factory function for Ray that returns a new wrapper instance per worker.
    # For parallel workers, Ray creates environments itself, but the wrapper needs
    # dependency injection, so a factory is required.
    def env_creator(config):
        local_env = GridEnv()
        return GridEnvRLlibWrapper(env=local_env)

    register_env(env_name, env_creator)
    print(f"   Environment '{env_name}' registered.")

    # --- 5. GET CONFIG ---
    print("Loading IPPO Configuration...")
    config = create_ippo_config(env_name=env_name)

    # --- 6. BUILD ALGORITHM, THEN TRAIN ---
    print("Building Algorithm...")
    algo = config.build_algo()

    print("Starting Training Loop...")
    try:
        for i in range(10):  # Training for 10 iterations as a test
            print(f"\n--- Iteration {i+1}/10 ---")
            result = algo.train()

            # Extracts key metrics from training results.
            # Keys may vary between RLlib versions - use .get() for safety
            episode_reward = result.get('env_runners', {}).get('episode_return_mean', 0.0)
            episode_len = result.get('env_runners', {}).get('episode_len_mean', 0.0)

            print(f"   Reward Mean: {episode_reward:.2f}")
            print(f"   Episode Len: {episode_len:.1f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Cleans up resources.
        algo.stop()
        ray.shutdown()
        print("\nTraining finished. Resources released.")


if __name__ == "__main__":
    main()