from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import (
    LSTMSequenceEncoder,
    SequenceEncoder,
    time_mask,
)


class GRUClassifier(nn.Module):
    """Causal prefix classifier, supervised by the entity's detection indicator.

    Each valid step predicts the same eventual observed label with BCE. Scores
    need not increase with time and are not survival hazards. Censored fraud is
    a negative under this target; true onset is never used for supervision.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.encoder = SequenceEncoder(
            input_dim,
            hidden_dim,
        )

        self.head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(self, x, lengths):
        h = self.encoder(
            x,
            lengths,
        )

        logits = self.head(h).squeeze(-1)

        return {
            "logits": logits,
            "risk": torch.sigmoid(logits),
        }

    def loss(self, batch):
        out = self(
            batch["x"],
            batch["lengths"],
        )

        mask = time_mask(
            batch["lengths"],
            batch["x"].shape[1],
        )

        # Every valid time step receives the entity's final observed label.
        target = (
            batch["detected"]
            .float()[:, None]
            .expand_as(out["logits"])
        )

        loss = F.binary_cross_entropy_with_logits(
            out["logits"][mask],
            target[mask],
        )

        return loss, out


class LSTMClassifier(GRUClassifier):
    """LSTM version of the causal prefix classifier."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(
            input_dim,
            hidden_dim,
        )

        self.encoder = LSTMSequenceEncoder(
            input_dim,
            hidden_dim,
        )


class SAFER(GRUClassifier):
    """Detection-time survival baseline using a GRU encoder.

    A head logit defines h(t), the conditional detection hazard.

    For detection at d:

        L = h(d) * product_{j<d}(1-h(j))

    For censoring at c:

        L = product_{j<=c}(1-h(j))

    Risk is the cumulative detection probability.
    """

    def forward(self, x, lengths):
        out = super().forward(
            x,
            lengths,
        )

        mask = time_mask(
            lengths,
            x.shape[1],
        )

        # log(1 - h_t)
        log_survival_increment = F.logsigmoid(
            -out["logits"]
        )

        # Padding contributes nothing.
        log_survival_increment = (
            log_survival_increment
            .masked_fill(~mask, 0)
        )

        # log S(t)
        log_survival = (
            log_survival_increment
            .cumsum(1)
        )

        # cumulative detection probability:
        # 1 - S(t)
        out["risk"] = -torch.expm1(
            log_survival
        )

        out["log_survival"] = log_survival

        return out

    def loss(self, batch):
        out = self(
            batch["x"],
            batch["lengths"],
        )

        detected = batch["detected"].bool()

        # If detected:
        #     endpoint = real detection time td
        #
        # If censored:
        #     endpoint = last observed time T
        end = torch.where(
            detected,
            batch["t_detect"],
            batch["lengths"] - 1,
        )

        if (
            (end < 0)
            | (end >= batch["lengths"])
        ).any():
            raise ValueError(
                "Detection endpoint must fall within the observed sequence"
            )

        survival = (
            out["log_survival"]
            .gather(
                1,
                end[:, None],
            )
            .squeeze(1)
        )

        logits = (
            out["logits"]
            .gather(
                1,
                end[:, None],
            )
            .squeeze(1)
        )

        # For detected entities:
        #
        # log likelihood
        # =
        # sum_{j<d} log(1-h_j)
        # +
        # log(h_d)
        #
        # Adding the endpoint logit replaces
        # log(1-h_d) with log(h_d).
        out["log_likelihood"] = (
            survival
            + torch.where(
                detected,
                logits,
                0,
            )
        )

        loss = -out[
            "log_likelihood"
        ].mean()

        return loss, out


class LSTMSAFER(SAFER):
    """Detection-time survival baseline using an LSTM encoder."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(
            input_dim,
            hidden_dim,
        )

        self.encoder = LSTMSequenceEncoder(
            input_dim,
            hidden_dim,
        )