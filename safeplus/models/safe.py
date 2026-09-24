from __future__ import annotations

import torch
from torch import nn

from .common import SequenceEncoder


class SAFE(nn.Module):
    """Discrete SAFE left-censoring/early-detection objective.

    S(t) = product_{j<=t}(1-h(j)). Detected entities contribute 1-S(t_d),
    rewarding event mass anywhere up to detection, rather than exactly at t_d.
    Censored entities contribute S(length-1). This one-head model does not
    separate commission from delayed detection or infer an onset posterior.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = SequenceEncoder(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        hazard = torch.sigmoid(self.head(self.encoder(x, lengths)).squeeze(-1))
        # This reference path uses probability products, unlike SAFE-r/SafePlus.
        # Long horizons can underflow survival or round risk to one. Investigate
        # log-space survival before treating flat scores as a modeling failure.
        # Raw padded risk is not masked here; Trainer and CLI predict carry the
        # last valid score forward before exposing predictions/metrics.
        survival = torch.cumprod(1.0 - hazard.clamp(max=1 - 1e-6), dim=1)
        return {"hazard": hazard, "survival": survival, "risk": 1.0 - survival}

    def loss(self, batch):
        out = self(batch["x"], batch["lengths"])
        eps = 1e-8
        losses = []
        for i in range(len(batch["x"])):
            end = int(batch["t_detect"][i] if batch["detected"][i] else batch["lengths"][i] - 1)
            surv = out["survival"][i, end]
            likelihood = (1.0 - surv) if batch["detected"][i] else surv
            # The floor bounds loss but also yields zero gradient below eps.
            # Any stabilization rewrite needs extreme-logit gradient tests.
            losses.append(-torch.log(likelihood.clamp_min(eps)))
        return torch.stack(losses).mean(), out
