import numpy as np
import pytest

from safeplus.evaluation.metrics import evaluate_predictions


def test_early_detection_excludes_on_time_late_and_missing_alarms():
    metrics = evaluate_predictions(
        y_true=np.array([1, 1, 1, 1, 0]),
        risk=np.array([
            [0.8, 0.8, 0.8],  # Early alarm.
            [0.1, 0.8, 0.8],  # Alarm at detection.
            [0.1, 0.1, 0.8],  # Late alarm.
            [0.1, 0.1, 0.1],  # No alarm.
            [0.8, 0.8, 0.8],  # False positive.
        ]),
        threshold=0.5,
        t_detect=np.array([1, 1, 1, 2, 2]),
    )
    assert metrics["early_detected_rate"] == pytest.approx(0.25)
    # Lead time still includes on-time and late true-positive alarms.
    assert metrics["mean_lead_time"] == pytest.approx(0.0)


def test_early_detection_without_positive_labels():
    metrics = evaluate_predictions(
        np.array([0]), np.array([[0.8, 0.9]]), 0.5, np.array([1])
    )
    assert metrics["early_detected_rate"] == 0.0
    assert np.isnan(metrics["mean_lead_time"])
