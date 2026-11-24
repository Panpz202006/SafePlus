"""Regression checks for the baseline split and evaluation contract."""

import unittest

import numpy as np

from safeplus.data import filter_legacy_overlap, fixed_prefixes
from safeplus.evaluation import prefix_metrics, validation_cutoff
from safeplus.training import length_batches


class EvaluationTests(unittest.TestCase):
    def test_overlap_filter_ignores_label_and_preserves_holdout(self):
        held_out = [(np.array([[0.0], [1.0]]), 1)]
        validation = [(np.array([[-0.0], [1.0]]), 0), (np.array([[2.0]]), 1)]
        training = [(np.array([[2.0]]), 0), (np.array([[3.0]]), 0)]
        train, valid, test = filter_legacy_overlap(training, validation, held_out)
        self.assertEqual([len(train), len(valid), len(test)], [1, 1, 1])
        self.assertIs(test, held_out)

    def test_fixed_prefix_hides_suffix_and_length(self):
        features, labels = fixed_prefixes(
            [(np.array([[1], [2], [3]]), 0), (np.array([[1], [2]]), 1)], 2
        )
        np.testing.assert_array_equal(features[0], features[1])
        np.testing.assert_array_equal(labels, [False, True])

    def test_length_batches_keep_remainders(self):
        records = [(np.zeros((2, 1)), 1)] * 3 + [(np.zeros((3, 1)), 0)]
        batches = length_batches(records, 2, seed=7)
        self.assertEqual(sum(len(labels) for _, labels in batches), 4)

    def test_cutoff_choice_and_prefix_metrics(self):
        labels = [True, False]
        scores = np.array([[0.8, 0.6], [0.9, 0.85]])
        accuracy, cutoff = validation_cutoff(scores, labels, [0.7, 0.8, 0.85])
        self.assertEqual((accuracy, cutoff), (1.0, 0.8))
        self.assertEqual(prefix_metrics(labels, scores <= cutoff), [(1.0,) * 4] * 2)
