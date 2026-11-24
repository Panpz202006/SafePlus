"""Training and validation selection for the audited SAFE baseline."""

from collections import defaultdict
import random

import numpy as np

from .evaluation import validation_cutoff


def length_batches(records, batch_size, *, seed):
    """Group by observed history length and retain every remainder batch."""
    buckets = defaultdict(list)
    for row, label in records:
        buckets[len(row)].append((row, label))
    batches = []
    for _, group in sorted(buckets.items()):
        for start in range(0, len(group), batch_size):
            features, labels = zip(*group[start : start + batch_size])
            batches.append((np.stack(features), np.asarray(labels)))
    random.Random(seed).shuffle(batches)
    return batches


def fit_baseline(train_batch, batches, validate, labels, thresholds, save):
    """Train and save using validation data only. Test inputs are absent."""
    best = (-1, 0, None)
    for epoch in range(1, 501):
        for features, events in batches:
            if not np.isfinite(train_batch(features, events)):
                raise ValueError(f"nonfinite training loss at epoch {epoch}")
        if epoch % 5 == 0:
            accuracy, cutoff = validation_cutoff(validate(), labels, thresholds)
            if accuracy > best[0]:
                best = (accuracy, epoch, cutoff)
                save()
                print("validation", best, flush=True)
        if epoch - best[1] > 20:
            break
    return best, epoch
