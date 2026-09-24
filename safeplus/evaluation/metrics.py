from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(fn, y, score):
    return float(fn(y, score)) if len(np.unique(y)) > 1 else float("nan")


def evaluate_predictions(
    y_true: np.ndarray,
    risk: np.ndarray,
    threshold: float,
    t_detect: np.ndarray | None = None,
    t_onset: np.ndarray | None = None,
    onset_posterior: np.ndarray | None = None,
) -> dict[str, float]:
    """Score carried-forward risks against observed detection labels.

    Callers must repeat the final valid risk through padding before calling.
    Early rate counts first alarms strictly before detection; lead time averages
    all observed-positive entities that ever alarm, including late alarms.
    Onset metrics require independently known in-range onset bins and use the
    observation-conditioned posterior, not the causal online score.
    """
    final_risk = risk[:, -1]
    pred = final_risk >= threshold
    result = {
        "auprc": _safe_metric(average_precision_score, y_true, final_risk),
        "auroc": _safe_metric(roc_auc_score, y_true, final_risk),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }
    if t_detect is not None:
        crossings = np.argmax(risk >= threshold, axis=1)
        # argmax returns zero even if there is no crossing; guard with any().
        crossed = (risk >= threshold).any(axis=1)
        positive = y_true.astype(bool) & crossed
        early = positive & (crossings < t_detect)
        result["early_detected_rate"] = float(early.sum() / max(1, y_true.sum()))
        result["mean_lead_time"] = (
            float(np.mean(t_detect[positive] - crossings[positive]))
            if positive.any()
            else float("nan")
        )
    if onset_posterior is not None and t_onset is not None:
        known = t_onset >= 0
        estimate = onset_posterior.argmax(axis=1)
        result["onset_mae"] = (
            float(np.abs(estimate[known] - t_onset[known]).mean()) if known.any() else float("nan")
        )
        if known.any():
            # Preserve no-commission mass for censored entities: renormalizing
            # onset bins would change this to a conditional-on-commission NLL.
            prob = onset_posterior[np.arange(len(t_onset))[known], t_onset[known]]
            result["onset_nll"] = float(-np.log(np.clip(prob, 1e-12, 1)).mean())
    return result


def prefix_metrics(y_true: np.ndarray, risk: np.ndarray, threshold: float, ks=(1, 2, 3, 4, 5)):
    rows = {}
    for k in ks:
        pred = risk[:, min(k, risk.shape[1]) - 1] >= threshold
        rows[f"f1@{k}"] = float(f1_score(y_true, pred, zero_division=0))
        rows[f"accuracy@{k}"] = float(accuracy_score(y_true, pred))
    return rows
