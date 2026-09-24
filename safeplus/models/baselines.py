from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import SequenceEncoder, time_mask


class GRUClassifier(nn.Module):
    """Causal prefix classifier, supervised by the entity's detection indicator.

    Each valid step predicts the same eventual observed label with BCE. Scores
    need not increase with time and are not survival hazards. Censored fraud is
    a negative under this target; true onset is never used for supervision.
    """
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
        # Reduce over valid time steps, not entities: longer histories receive
        # more weight. Preserve this distinction when comparing survival losses,
        # which average one log likelihood per entity.
        target = batch["detected"].float()[:, None].expand_as(out["logits"])
        loss = F.binary_cross_entropy_with_logits(out["logits"][mask], target[mask])
        return loss, out


class SAFER(GRUClassifier):
    """Detection-time survival baseline; registry name: ``safe-r``.

    A head logit defines h(t), the conditional detection hazard. For detection
    at d, L = h(d) * product_{j<d}(1-h(j)); for censoring at c,
    L = product_{j<=c}(1-h(j)). Risk is the cumulative detection probability.
    The inherited classifier head is reused, but its BCE objective is not.
    """

    def forward(self, x, lengths):
        out = super().forward(x, lengths)
        mask = time_mask(lengths, x.shape[1])
        # Zero log-survival increments make padding multiplicatively neutral.
        # logsigmoid avoids log(sigmoid(...)) saturation; -expm1(s) accurately
        # computes 1-exp(s) when cumulative risk is close to zero.
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
