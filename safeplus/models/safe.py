from __future__ import annotations

import torch
from torch import nn

from .common import (
    LSTMSequenceEncoder,
    SequenceEncoder,
)


class SAFE(nn.Module):
    """Discrete SAFE fraud-onset-only objective.

    S_f(t) = product_{j<=t}(1-h_f(j))

    For detected entities:
        likelihood = 1 - S_f(t_d)
        meaning fraud must have occurred no later than detection.

    For censored entities:
        likelihood = S_f(T)
        meaning fraud is treated as not having occurred before
        the observation boundary.

    This one-head model does not separately model detection hazard
    and does not infer a posterior over the latent onset time.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.encoder = SequenceEncoder(
            input_dim,
            hidden_dim,
        )

        self.head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        x,
        lengths,
    ):
        hidden = self.encoder(
            x,
            lengths,
        )

        logits = self.head(
            hidden
        ).squeeze(-1)

        # Discrete fraud-onset hazard h_f(t)
        hazard = torch.sigmoid(
            logits
        )

        # S_f(t) = product_{j<=t}(1-h_f(j))
        survival = torch.cumprod(
            1.0 - hazard.clamp(
                max=1 - 1e-6
            ),
            dim=1,
        )

        # P(t_f <= t)
        risk = 1.0 - survival

        return {
            "logits": logits,
            "hazard": hazard,
            "survival": survival,
            "risk": risk,
        }

    def loss(
        self,
        batch,
    ):
        out = self(
            batch["x"],
            batch["lengths"],
        )

        eps = 1e-8

        losses = []

        for i in range(
            len(batch["x"])
        ):
            detected = bool(
                batch["detected"][i]
            )

            if detected:
                # Known detection time td.
                end = int(
                    batch["t_detect"][i]
                )

                # SAFE only knows:
                #
                # tf <= td
                #
                # Therefore maximize:
                #
                # P(tf <= td)
                # = 1 - S_f(td)
                survival_at_end = (
                    out["survival"][i, end]
                )

                likelihood = (
                    1.0
                    - survival_at_end
                )

            else:
                # No detection observed before
                # the end of the sequence.
                end = int(
                    batch["lengths"][i] - 1
                )

                # SAFE treats this case as:
                #
                # tf > T
                #
                # Therefore maximize:
                #
                # P(tf > T)
                # = S_f(T)
                survival_at_end = (
                    out["survival"][i, end]
                )

                likelihood = (
                    survival_at_end
                )

            loss_i = -torch.log(
                likelihood.clamp_min(
                    eps
                )
            )

            losses.append(
                loss_i
            )

        loss = torch.stack(
            losses
        ).mean()

        return loss, out


class LSTMSAFE(SAFE):
    """LSTM version of the SAFE fraud-onset-only baseline.

    The SAFE objective is unchanged. Only the sequence encoder
    is replaced from GRU to LSTM so that GRU-SAFE and LSTM-SAFE
    form a controlled comparison.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
        )

        self.encoder = (
            LSTMSequenceEncoder(
                input_dim,
                hidden_dim,
            )
        )