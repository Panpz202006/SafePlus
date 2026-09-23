from __future__ import annotations

import numpy as np


def generate_synthetic(
    n: int = 2000,
    length: int = 24,
    features: int = 8,
    fraud_rate: float = 0.35,
    censor_rate: float = 0.20,
    delay_mean: float = 4.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Generate sequences with known onset and delayed/no detection.

    Covariates shift after onset. The action time is retained in `t_onset` only
    for evaluation and is not consumed by the model.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n, length, features)).astype(np.float32)
    is_fraud = rng.random(n) < fraud_rate
    onset = np.full(n, -1, dtype=np.int64)
    detected = np.zeros(n, dtype=np.int64)
    t_detect = np.full(n, length - 1, dtype=np.int64)
    signal = rng.normal(0.9, 0.15, size=features).astype(np.float32)

    for i in np.flatnonzero(is_fraud):
        onset[i] = rng.integers(max(2, length // 6), max(3, 2 * length // 3))
        ramp = np.linspace(0.25, 1.0, length - onset[i], dtype=np.float32)[:, None]
        x[i, onset[i] :] += ramp * signal
        delay = max(1, int(rng.exponential(delay_mean)))
        candidate = onset[i] + delay
        if candidate < length and rng.random() >= censor_rate:
            detected[i] = 1
            t_detect[i] = candidate

    return {
        "x": x,
        "lengths": np.full(n, length, dtype=np.int64),
        "detected": detected,
        "t_detect": t_detect,
        "t_onset": onset,
        "entity_id": np.arange(n, dtype=np.int64),
    }
