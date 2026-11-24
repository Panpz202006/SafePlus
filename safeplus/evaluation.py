"""Shared fixed-prefix scoring for SAFE, SAFEPlus, and comparators."""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def validation_cutoff(scores, event_labels, thresholds):
    """Select by mean prefix accuracy. Grid order breaks ties."""
    labels = np.asarray(event_labels, dtype=bool)[:, None]
    accuracy = [((scores <= cutoff) == labels).mean() for cutoff in thresholds]
    index = int(np.argmax(accuracy))
    return accuracy[index], thresholds[index]


def prefix_metrics(event_labels, event_predictions):
    """Return precision, recall, F1, and accuracy by prediction timestamp.

    The positive class is the observed event. Each column is a prediction
    from that prefix alone, and rows must match the frozen evaluation cohort.
    """
    labels = np.asarray(event_labels, dtype=bool)
    predictions = np.asarray(event_predictions, dtype=bool)
    if predictions.ndim != 2 or labels.shape != (len(predictions),):
        raise ValueError("expected one event label per row of prefix predictions")
    metrics = []
    for prefix in predictions.T:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, prefix, average="binary", zero_division=0
        )
        metrics.append((precision, recall, f1, float((prefix == labels).mean())))
    return metrics
