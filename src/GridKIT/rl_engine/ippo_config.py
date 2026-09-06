# rl_engine/ippo_config.py
# ─────────────────────────────────────────────────────────────────────────────
# IPPO (Independent PPO) Configuration for GridKIT.
#
# Design Decisions:
# - Multi-Agent with one shared policy PER DEVICE TYPE: all EV agents share
#   "ev_policy", all battery agents share "battery_policy", all heat-pump
#   agents share "hp_policy" (pooling experience within a device type, the way
#   a single shared policy pools it across households). One policy across ALL
#   device types would not work: EV/battery have 3 discrete actions, heat pump
#   only 2 (OFF/HEAT), and the three devices act on physically different
#   quantities — see constants.ACTION_DIM_BY_DEVICE and
#   GridEnvRLlibWrapper.action_spaces, which already give each device type its
#   own action space; this must route to a matching per-type policy.
# - Discrete Action Space: each PolicySpec's action_space is explicit (not
#   inferred), since the wrapper's action_space is heterogeneous (a per-agent
#   Dict), so RLlib cannot infer a single space to hand every policy.
# ─────────────────────────────────────────────────────────────────────────────

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
import gymnasium as gym

from GridKIT.core import constants as const
from GridKIT.core.config import settings
from GridKIT.core.models import device_of


def _policy_id(agent_id: str, *args, **kwargs) -> str:
    """Route an agent to its device type's shared policy (e.g. "h3::hp" -> "hp_policy")."""
    return f"{device_of(agent_id)}_policy"


_OBS_SPACE = gym.spaces.Box(low=0.0, high=1.0, shape=(const.OBS_DIM_MULTIDEVICE,))


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
    return (
        PPOConfig()
        .environment(env=env_name)
        .env_runners(num_env_runners=num_env_runners)
        .multi_agent(
            policies={
                f"{device}_policy": PolicySpec(
                    observation_space=_OBS_SPACE,
                    action_space=gym.spaces.Discrete(dim),
                )
                for device, dim in const.ACTION_DIM_BY_DEVICE.items()
            },
            # Maps each agent to its device type's shared policy (see _policy_id).
            policy_mapping_fn=_policy_id,
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