"""
Adapter that exposes the trained per-device RLlib policies as a scenarios.Policy,
so the selfish congestion-aware agents (scenario 3) run through the *same*
scenarios.runner as the non-RL behaviours and produce comparable EpisodeMetrics.

Multi-device: there is one shared policy per device type (ev / battery / hp). The
adapter holds all three RLModules, routes each agent's observation to the module
for its device type, and returns a per-device action *index*.

By default it SAMPLES from each policy (independent draw per agent) rather than
taking the argmax: a deterministic shared policy would re-synchronize identical
agents, whereas independent sampling lets them scatter (mixed-strategy
desynchronization).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from GridKIT.core.models import Observation, device_of
from GridKIT.rl_engine.obs_norm import normalize_observation_multidevice


class RLlibPolicyAdapter:
    def __init__(self, modules: dict, sample: bool = True, seed: int = 0):
        """modules: {device_type -> trained RLModule} (e.g. Trainer.get_policy_modules())."""
        self._modules = modules
        self._sample = sample
        self._gen = torch.Generator().manual_seed(seed)

    def reset(self, day_ahead_prices: list[float], rng: np.random.Generator) -> None:
        # tie action sampling to the episode seed for reproducibility
        self._gen = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))

    def act(self, observations: dict[str, Observation]) -> dict[str, int]:
        # group agents by device type → one batched forward pass per device policy
        by_device: dict[str, list[str]] = defaultdict(list)
        for aid in observations:
            by_device[device_of(aid)].append(aid)

        actions: dict[str, int] = {}
        for device, aids in by_device.items():
            module = self._modules[device]
            batch = np.stack([normalize_observation_multidevice(observations[a]) for a in aids])
            with torch.no_grad():
                out = module.forward_inference({"obs": torch.as_tensor(batch, dtype=torch.float32)})
            logits = out["action_dist_inputs"]
            if self._sample:
                probs = torch.softmax(logits, dim=-1)
                idx = torch.multinomial(probs, num_samples=1, generator=self._gen).squeeze(-1)
            else:
                idx = torch.argmax(logits, dim=-1)
            for i, a in enumerate(aids):
                actions[a] = int(idx[i])
        return actions
