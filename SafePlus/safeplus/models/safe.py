from __future__ import annotations

import torch
from torch import nn

from .common import SequenceEncoder


class SAFE(nn.Module):
    """Discrete SAFE: early-detection likelihood F(t_d) for detected entities."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = SequenceEncoder(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        hazard = torch.sigmoid(self.head(self.encoder(x, lengths)).squeeze(-1))
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
            losses.append(-torch.log(likelihood.clamp_min(eps)))
        return torch.stack(losses).mean(), out
