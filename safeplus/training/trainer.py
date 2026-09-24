from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score


class Trainer:
    """Optimize model-specific losses; select checkpoints by detection-label AP.

    SafePlus commission risk is ranked against observed detection indicators,
    not independently observed fraud status. Loss and selection metric can
    therefore improve at different times. Inspect both curves before tuning.
    """
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=0.0):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def _move(self, batch):
        return {key: value.to(self.device) for key, value in batch.items()}

    def run_epoch(self, loader, train=True):
        self.model.train(train)
        losses, risks, labels, detects, onsets, posteriors = [], [], [], [], [], []
        for raw in loader:
            batch = self._move(raw)
            with torch.set_grad_enabled(train):
                loss, out = self.model.loss(batch)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Clip the global gradient norm. When debugging instability,
                    # inspect pre-clip norms and each hazard head's gradients.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()
            losses.append((float(loss.detach()), len(batch["x"])))
            risk = out["risk"].detach().clone()
            # Carry the last valid prediction across padding for all models.
            steps = torch.arange(risk.shape[1], device=risk.device)[None, :]
            final = risk.gather(1, (batch["lengths"] - 1)[:, None])
            risk = torch.where(steps < batch["lengths"][:, None], risk, final)
            risks.append(risk.cpu().numpy())
            labels.append(batch["detected"].cpu().numpy().astype(int))
            detects.append(batch["t_detect"].cpu().numpy())
            onsets.append(batch["t_onset"].cpu().numpy())
            if "onset_posterior" in out:
                posteriors.append(out["onset_posterior"].detach().cpu().numpy())
        payload = {
            "loss": float(sum(loss * n for loss, n in losses) / sum(n for _, n in losses)),
            "risk": np.concatenate(risks),
            "labels": np.concatenate(labels),
            "t_detect": np.concatenate(detects),
            "t_onset": np.concatenate(onsets),
        }
        if posteriors:
            payload["onset_posterior"] = np.concatenate(posteriors)
        return payload

    def fit(self, train_loader, val_loader, epochs=30, patience=7):
        best_score, best_state, stale = -np.inf, None, 0
        history = []
        for epoch in range(epochs):
            train = self.run_epoch(train_loader, train=True)
            val = self.run_epoch(val_loader, train=False)
            score = average_precision_score(val["labels"], val["risk"][:, -1])
            # Strict improvement resets patience; ties consume it. A slow-starting
            # model can restore an early checkpoint despite falling NLL. Tune
            # patience/selection on validation only, keeping test results held out.
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train["loss"],
                    "val_loss": val["loss"],
                    "val_auprc": float(score),
                }
            )
            if score > best_score:
                best_score, best_state, stale = score, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history
