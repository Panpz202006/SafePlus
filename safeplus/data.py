"""Input preparation for the exploratory legacy SAFE evaluation."""

from itertools import combinations

import numpy as np


def legacy_records(features, lengths, events, feature_count):
    return [
        (np.asarray(row, dtype=np.float32).reshape(int(length), feature_count), event)
        for row, length, event in zip(features, lengths, events, strict=True)
    ]


def filter_legacy_overlap(train, validation, test):
    """Preserve the holdout and exclude exact cross-split feature matches.

    The archived arrays lack account IDs. Feature equality is the only
    overlap criterion enforced here, and event labels play no role.
    """
    def signature(row):
        return np.where(row == 0, np.float32(0), row).tobytes()

    def keys(records):
        return {signature(row) for row, _ in records}

    def exclude(records, blocked):
        return [(row, event) for row, event in records if signature(row) not in blocked]

    test_keys = keys(test)
    filtered = (
        exclude(train, test_keys | keys(validation)),
        exclude(validation, test_keys),
        test,
    )
    if any(left & right for left, right in combinations(map(keys, filtered), 2)):
        raise ValueError("exact feature overlap across splits")
    return filtered


def fixed_prefixes(records, horizon):
    """Pass only the observed prefix, without the final history length."""
    if any(len(row) < horizon for row, _ in records):
        raise ValueError("legacy evaluation requires every record to reach the horizon")
    features = np.stack([row[:horizon] for row, _ in records])
    labels = np.asarray([event for _, event in records], dtype=bool)
    return features, labels
