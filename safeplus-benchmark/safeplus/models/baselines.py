from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import SequenceEncoder, time_mask


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = SequenceEncoder(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        logits = self.head(self.encoder(x, lengths)).squeeze(-1)
        return {"logits": logits, "risk": torch.sigmoid(logits)}

    def loss(self, batch):
        out = self(batch["x"], batch["lengths"])
        mask = time_mask(batch["lengths"], batch["x"].shape[1])
        target = batch["detected"].float()[:, None].expand_as(out["logits"])
        loss = F.binary_cross_entropy_with_logits(out["logits"][mask], target[mask])
        return loss, out


class SAFER(GRUClassifier):
    """Ordinary detection-time survival likelihood (draft Section 2.1)."""

    def forward(self, x, lengths):
        out = super().forward(x, lengths)
        mask = time_mask(lengths, x.shape[1])
        log_survival = F.logsigmoid(-out["logits"]).masked_fill(~mask, 0).cumsum(1)
        out["risk"] = -torch.expm1(log_survival)
        out["log_survival"] = log_survival
        return out

    def loss(self, batch):
        out = self(batch["x"], batch["lengths"])
        detected = batch["detected"].bool()
        end = torch.where(detected, batch["t_detect"], batch["lengths"] - 1)
        if ((end < 0) | (end >= batch["lengths"])).any():
            raise ValueError("Detection endpoint must fall within the observed sequence")
        survival = out["log_survival"].gather(1, end[:, None]).squeeze(1)
        logits = out["logits"].gather(1, end[:, None]).squeeze(1)
        # Replacing log(1-h) at the endpoint with log(h) adds the logit.
        out["log_likelihood"] = survival + torch.where(detected, logits, 0)
        return -out["log_likelihood"].mean(), out
