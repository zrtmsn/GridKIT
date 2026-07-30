# rl_engine/ippo_config.py
# ─────────────────────────────────────────────────────────────────────────────
# IPPO (Independent PPO) Configuration for GridKIT.
#
# Design Decisions:
# - One shared policy PER device type (ev_policy / battery_policy / hp_policy):
#   every agent of a given device type shares neural-network weights, pooling
#   experience across households while keeping each device's action space small
#   (agent-per-device avoids the combinatorial joint action space).
# - Discrete Action Space, per device: ev/battery = Discrete(3), hp = Discrete(2),
#   set explicitly on each PolicySpec. policy_mapping_fn routes by agent-id suffix.
# ─────────────────────────────────────────────────────────────────────────────

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from GridKIT.core import constants as const
from GridKIT.core.config import settings
from GridKIT.core.models import device_of


def _policy_name(device: str) -> str:
    return f"{device}_policy"


def policy_mapping_fn(agent_id, *args, **kwargs):
    """Route each agent to the shared policy for its device type (agent_id = bus::device)."""
    return _policy_name(device_of(agent_id))


def create_ippo_config(
    env_name: str = settings.rllib_env_registry_name,
    num_env_runners: int = settings.ippo_num_env_runners,
    train_batch_size: int = settings.ippo_train_batch_size,
    minibatch_size: int = settings.ippo_minibatch_size,
    num_epochs: int = settings.ippo_n_epochs,
    lr: float = settings.ippo_learning_rate,
    gamma: float = settings.ippo_gamma,
    lambda_: float = settings.ippo_gae_lambda,
    clip_param: float = settings.ippo_clip_eps,
    entropy_coeff: float = settings.ippo_entropy_coeff,
    evaluation_interval: int = settings.ippo_evaluation_interval,
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
    # One shared policy PER device type (ev / battery / hp). Obs space is the
    # uniform 10-dim Box; the action space differs by device type, so it is set
    # explicitly on each PolicySpec.
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(const.OBS_DIM_MULTIDEVICE,), dtype=np.float32)
    policies = {
        _policy_name(dev): PolicySpec(
            observation_space=obs_space,
            action_space=gym.spaces.Discrete(const.ACTION_DIM_BY_DEVICE[dev]),
        )
        for dev in const.CONTROLLABLE_DEVICE_TYPES
    }

    return (
        PPOConfig()
        .environment(env=env_name)
        .env_runners(num_env_runners=num_env_runners)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
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
            evaluation_num_env_runners=settings.ippo_num_evaluation_env_runners,
            evaluation_interval=evaluation_interval,
        )
    )