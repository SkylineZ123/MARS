"""Execute the controller selected by MARS using a shared frozen RL Token."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from ..router import pack_modalities


@dataclass
class PolicyStep:
    action: np.ndarray
    active_name: str
    switched: bool
    probabilities: np.ndarray


class RoutedRLTPolicy:
    """One environment action per call, with immediate queue invalidation on handoff.

    reference(features, normalized_q) -> normalized generalist chunk (H, d_a).
    It may reuse VLA KV caches held by the caller's features object. All specialist
    chunks use that same action normalization; unnormalize after this policy returns.
    """

    def __init__(self, controller, adapter, reference, specialists: dict, query_interval: int = 1):
        if query_interval <= 0:
            raise ValueError("query_interval must be positive")
        expected = set(controller.classes[1:])
        if set(specialists) != expected:
            raise ValueError(f"specialists must match router candidates {sorted(expected)}")
        self.controller, self.adapter = controller, adapter.eval()
        self.reference, self.specialists = reference, dict(specialists)
        self.query_interval = query_interval
        self.reset()

    def reset(self):
        self.controller.reset()
        self.queue = deque()
        self.t = 0
        self.probabilities = np.eye(len(self.controller.classes), dtype=np.float32)[0]

    @torch.no_grad()
    def step(self, features, q, explore=False, rng=None) -> PolicyStep:
        if features.hidden.shape[0] != 1:
            raise ValueError("streaming deployment requires a single observation")
        z = self.adapter.encode_features(features)
        q = np.asarray(q, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError("normalized proprioception q must be a vector")
        switched = False
        if self.t % self.query_interval == 0:
            packed = (pack_modalities(features, z.to(features.hidden.device))[0]
                      if getattr(self.controller.router, "needs_modalities", False) else z[0]).cpu().numpy()
            _, switched, self.probabilities = self.controller.query(packed, q)
            if switched:
                self.queue.clear()
        if not self.queue:
            ref = np.asarray(self.reference(features, q), dtype=np.float32)
            if ref.ndim != 2 or len(ref) == 0 or not np.isfinite(ref).all():
                raise ValueError("reference must return a finite nonempty (H, action_dim) chunk")
            name = self.controller.active_name
            if self.controller.active == 0:
                chunk = ref
            else:
                specialist = self.specialists[name]
                if len(ref) < specialist.C:
                    raise ValueError(f"reference horizon {len(ref)} is shorter than {name} C={specialist.C}")
                chunk = specialist.act(z[0].cpu().numpy(), q, ref[:specialist.C], explore=explore, rng=rng)
            if np.asarray(chunk).ndim != 2 or chunk.shape[1] != ref.shape[1] or len(chunk) == 0 or not np.isfinite(chunk).all():
                raise ValueError("controller returned an invalid action chunk")
            self.queue.extend(np.asarray(chunk, dtype=np.float32).copy())
        self.t += 1
        return PolicyStep(self.queue.popleft().copy(), self.controller.active_name, switched,
                          self.probabilities.copy())
