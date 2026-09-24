from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import SequenceEncoder, time_mask


class SafePlus(nn.Module):
    """Discrete joint commission/detection model (draft Sections 2.3 and 3).

    Within a bin, commission precedes detection, allowing same-bin detection.
    Exact inference is linear in batch size times padded sequence length.
    Both heads consume the same causal GRU state. Detection logits depend on
    calendar bin and history, not explicitly on onset k or elapsed delay t-k;
    onset only determines when detection survival starts. This restriction
    enables the suffix-sum inference below.
    """

    def __init__(self, input_dim: int, hidden_dim: int, inference="exact", em_samples=1):
        super().__init__()
        if inference not in {"exact", "stochastic_em"}:
            raise ValueError("inference must be exact or stochastic_em")
        if not isinstance(em_samples, int) or em_samples < 1:
            raise ValueError("em_samples must be a positive integer")
        self.inference = inference
        self.em_samples = em_samples
        self.encoder = SequenceEncoder(input_dim, hidden_dim)
        self.commission_head = nn.Linear(hidden_dim, 1)
        self.detection_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _event_mass(hazard):
        before = torch.cat([hazard.new_ones(1), (1 - hazard).cumprod(0)[:-1]])
        return before * hazard

    @staticmethod
    def _detect_at(hd, onset, detected_at):
        if detected_at < onset:
            return hd.new_zeros(())
        return (1 - hd[onset:detected_at]).prod() * hd[detected_at]

    @staticmethod
    def _undetected_through(hd, onset, horizon):
        return (1 - hd[onset : horizon + 1]).prod()

    def forward(self, x, lengths):
        h = self.encoder(x, lengths)
        lf = self.commission_head(h).squeeze(-1)
        ld = self.detection_head(h).squeeze(-1)
        mask = time_mask(lengths, x.shape[1])
        # Online risk integrates commission hazards only. Unlike the posterior
        # in loss(), it cannot use a future detection/censoring observation.
        log_sf = F.logsigmoid(-lf).masked_fill(~mask, 0).cumsum(1)
        return {
            "commission_logits": lf,
            "detection_logits": ld,
            "commission_hazard": lf.sigmoid().masked_fill(~mask, 0),
            "detection_hazard": ld.sigmoid().masked_fill(~mask, 0),
            "risk": -torch.expm1(log_sf),
        }

    @staticmethod
    def _log_joint(log_hf, log_sf, log_hd, log_sd, detected, endpoints):
        """Unnormalized joint log masses [B,L+1] for the observed endpoint.

        Inputs log_h* / log_s* are per-bin log(h) / log(1-h), not cumulative.
        For onset k and detection d:
          log p(k,d) = sum_{j<k} log_sf[j] + log_hf[k]
                       + sum_{k<=j<d} log_sd[j] + log_hd[d].
        For censoring at c, include detection survival through c and omit the
        detection event factor. The last column is survival of commission
        through c; it is impossible (-inf) for a detected entity.
        """
        steps = torch.arange(log_hf.shape[1], device=log_hf.device)[None, :]
        valid = steps <= endpoints[:, None]
        # Reverse cumulative sums avoid subtracting large nearly equal sums.
        detection_survival = (steps < endpoints[:, None]) | (valid & ~detected[:, None])
        # Include endpoint survival only for censoring. At same-bin detection
        # (k=d), the detection-survival product is empty and therefore one.
        terms = log_sd.masked_fill(~detection_survival, 0)
        suffix = terms.flip(1).cumsum(1).flip(1)
        before = torch.cat([log_sf.new_zeros((len(log_sf), 1)), log_sf[:, :-1]], 1)
        before = before.cumsum(1)
        # Shift before summation: commission survival ends at k-1, not k.
        event = log_hd.gather(1, endpoints[:, None]).squeeze(1)
        joint = log_hf + before + suffix + torch.where(detected, event, 0)[:, None]
        joint = joint.masked_fill(~valid, -torch.inf)
        # Impossible onset bins carry exactly zero posterior mass. Do not use
        # a finite sentinel here: it would admit events after the endpoint.
        no_commission = log_sf.masked_fill(~valid, 0).sum(1)
        no_commission = no_commission.masked_fill(detected, -torch.inf)
        return torch.cat([joint, no_commission[:, None]], 1)

    def likelihood_and_posterior(self, hf, hd, detected: bool, endpoint: int):
        """Compatibility helper; posterior excludes the no-commission state.

        An impossible observation has zero likelihood and an all-zero posterior.
        Training uses logits directly to avoid saturated probability arithmetic.
        """
        if hf.ndim != 1 or hd.shape != hf.shape or not 0 <= endpoint < len(hf):
            raise ValueError("Expected matching hazard vectors and an in-range endpoint")
        if ((hf < 0) | (hf > 1) | (hd < 0) | (hd > 1)).any():
            raise ValueError("Hazards must lie in [0, 1]")
        hf, hd = hf[: endpoint + 1], hd[: endpoint + 1]
        joint = self._log_joint(
            hf.log()[None],
            torch.log1p(-hf)[None],
            hd.log()[None],
            torch.log1p(-hd)[None],
            torch.tensor([detected], device=hf.device),
            torch.tensor([endpoint], device=hf.device),
        )[0]
        log_likelihood = torch.logsumexp(joint, 0)
        if torch.isneginf(log_likelihood):
            return log_likelihood.exp(), torch.zeros_like(hf)
        return log_likelihood.exp(), (joint[:-1] - log_likelihood).exp()

    @staticmethod
    def sample_onsets(posterior, samples=1):
        """Sample full posterior states; final column means onset after horizon."""
        # Detach the E-step distribution: gradients flow through selected joint
        # log masses in the M-step, not through the categorical sampling law.
        return torch.multinomial(posterior.detach(), samples, replacement=True)

    def loss(self, batch):
        out = self(batch["x"], batch["lengths"])
        detected = batch["detected"].bool()
        endpoints = torch.where(detected, batch["t_detect"], batch["lengths"] - 1)
        if ((endpoints < 0) | (endpoints >= batch["lengths"])).any():
            raise ValueError("Detection endpoint must fall within the observed sequence")
        lf, ld = out["commission_logits"], out["detection_logits"]
        joint = self._log_joint(
            F.logsigmoid(lf),
            F.logsigmoid(-lf),
            F.logsigmoid(ld),
            F.logsigmoid(-ld),
            detected,
            endpoints,
        )
        log_likelihood = torch.logsumexp(joint, 1)
        # Normalize across all L+1 states. For censored entities onset-only
        # probabilities need not sum to one: the remaining mass is no commission.
        posterior = (joint - log_likelihood[:, None]).exp()
        out.update(
            log_likelihood=log_likelihood,
            latent_posterior=posterior,
            onset_posterior=posterior[:, :-1],
            no_commission_posterior=posterior[:, -1],
        )
        if self.training and self.inference == "stochastic_em":
            # Monte Carlo complete-data objective; increasing samples reduces
            # sampling noise but does not avoid the full exact posterior above.
            # Its loss value is not directly comparable to marginal NLL.
            samples = self.sample_onsets(posterior, self.em_samples)
            loss = -joint.gather(1, samples).mean()
        else:
            loss = -log_likelihood.mean()
        return loss, out
