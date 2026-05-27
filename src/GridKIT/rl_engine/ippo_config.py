# rl_engine/ippo_config.py
# ─────────────────────────────────────────────────────────────────────────────
# IPPO (Independent PPO) Configuration for GridKIT.
#
# Design Decisions:
# - Multi-Agent with Shared Policy: All household agents use the same neural
#   network weights ("household_policy"). This accelerates convergence by pooling
#   experiences from all agents.
# - Discrete Action Space: Uses discrete actions (OFF/HALF/FULL) via the
#   GridEnvRLlibWrapper. The action space in the wrapper must match.
# ─────────────────────────────────────────────────────────────────────────────

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from GridKIT.core import constants as const


def create_ippo_config(
    env_name: str = const.RLLIB_ENV_REGISTRY_NAME,
    num_env_runners: int = const.IPPO_NUM_ENV_RUNNERS,
    train_batch_size: int = const.IPPO_TRAIN_BATCH_SIZE,
    minibatch_size: int = const.IPPO_MINIBATCH_SIZE,      
    num_epochs: int = const.IPPO_N_EPOCHS,
    lr: float = const.IPPO_LEARNING_RATE,
    gamma: float = const.IPPO_GAMMA,
    lambda_: float = const.IPPO_GAE_LAMBDA,
    clip_param: float = const.IPPO_CLIP_EPS,
    entropy_coeff: float = const.IPPO_ENTROPY_COEFF,
    evaluation_interval: int = const.IPPO_EVALUATION_INTERVAL,
) -> PPOConfig:
    """
    Creates and returns a configured PPOConfig for IPPO training.

    Args:
        env_name: Registered name of the environment.
        num_env_runners: Number of parallel environment workers.
        train_batch_size: Total steps collected before one update cycle.
        minibatch_size: Size of batches used for SGD updates within an epoch.
        num_epochs: Number of passes over the collected data per update.
        lr: Learning rate for the optimizer.
        gamma: Discount factor for future rewards.
        lambda_: GAE lambda parameter for advantage estimation.
        clip_param: PPO clipping range to prevent large policy updates.
        entropy_coeff: Bonus weight to encourage exploration.
        evaluation_interval: Run evaluation every N iterations.

    Returns:
        A configured ray.rllib.algorithms.ppo.PPOConfig object.
    """
    return (
        PPOConfig()
        .environment(env=env_name)
        .env_runners(num_env_runners=num_env_runners)
        .multi_agent(
            policies={
                # All agents map to this single shared policy
                "household_policy": PolicySpec(
                    observation_space=None,  # Inferred from environment
                    action_space=None,         # Inferred from environment
                )
            },
            # Maps every agent_id to the same policy
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "household_policy",
        )
        .training(
            lr=lr,
            gamma=gamma,
            lambda_=lambda_,
            clip_param=clip_param,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            entropy_coeff=entropy_coeff,
        )
        .evaluation(
            evaluation_num_env_runners=const.IPPO_NUM_EVALUATION_ENV_RUNNERS,
            evaluation_interval=evaluation_interval,
        )
    )