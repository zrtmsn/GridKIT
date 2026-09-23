# test_run_training_through_trainer.py
"""
Simple test script that uses the RLTrainer abstraction.
"""

import sys
import os
from pathlib import Path

# --- SETUP PATHS ---
script_dir = Path(__file__).parent  # src/GridKIT/
src_dir = script_dir.parent          # src/
sys.path.insert(0, str(src_dir))

# Changes working directory to project root for relative file access.
project_root = script_dir.parent.parent
os.chdir(project_root)

# Sets PYTHONPATH for all Ray workers.
os.environ["PYTHONPATH"] = f"{script_dir}{os.pathsep}{src_dir}"
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"


from GridKIT.grid_model.environment import GridEnv
from GridKIT.rl_engine import GridEnvRLlibWrapper, Trainer, create_ippo_config
from GridKIT.training_utils import force_always_connected


def main():
    # Factory: Creates a fresh Env+Wrapper for each Ray worker
    def my_env_factory(config=None):
        env = GridEnv()
        # ! TODO: remove force_always_connected once RL is adjusted to handle the new EV availability model
        force_always_connected(env)
        return GridEnvRLlibWrapper(env=env)

    trainer = Trainer(env_factory=my_env_factory, config_func=create_ippo_config)
    results = trainer.run()                                                  
    print("Training done! Received results for", len(results), "episodes.")


if __name__ == "__main__":
    main()

    


#TODO check if this script is obsolete now
