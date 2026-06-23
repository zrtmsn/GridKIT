# rl_engine/rl_policy.py
"""
Adapter that exposes a trained RLlib policy as a scenarios.Policy, so the
selfish congestion-aware agent (scenario 3) runs through the *same*
scenarios.runner as the non-RL behaviours and produces comparable
EpisodeMetrics (curtailment events, SoC satisfaction, ...).

By default it SAMPLES from the policy (independent draw per agent) rather than
taking the argmax: a deterministic shared policy would re-synchronize identical
agents, whereas independent sampling lets them scatter (the mixed-strategy
desynchronization discussed in the project design).
"""
from __future__ import annotations

import numpy as np
import torch

from GridKIT.core.models import ChargingAction, Observation
from GridKIT.rl_engine.obs_norm import normalize_observation


class RLlibPolicyAdapter:
    def __init__(self, module, sample: bool = True, seed: int = 0):
        self._module = module
        self._sample = sample
        self._gen = torch.Generator().manual_seed(seed)

    def reset(self, day_ahead_prices: list[float], rng: np.random.Generator) -> None:
        # tie action sampling to the episode seed for reproducibility
        self._gen = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))

    def act(self, observations: dict[str, Observation]) -> dict[str, ChargingAction]:
        agent_ids = list(observations.keys())
        batch = np.stack([normalize_observation(observations[a]) for a in agent_ids])
        with torch.no_grad():
            out = self._module.forward_inference({"obs": torch.as_tensor(batch, dtype=torch.float32)})
        logits = out["action_dist_inputs"]
        if self._sample:
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1, generator=self._gen).squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return {a: ChargingAction(int(idx[i])) for i, a in enumerate(agent_ids)}
