# -*- coding: utf-8 -*-
"""Tests for MatrixProfile."""

import os
import sys
import unittest

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.models.ts_matrix_profile import MatrixProfile
from pyod.utils.data import generate_ts_data


class TestMatrixProfile(unittest.TestCase):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            generate_ts_data(n_train=300, n_test=100, contamination=0.05,
                             random_state=42)

    def test_fit(self):
        clf = MatrixProfile(window_size=20, contamination=0.1)
        clf.fit(self.X_train)
        assert len(clf.decision_scores_) == 300
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'threshold_')

    def test_multivariate(self):
        X_multi = generate_ts_data(
            n_train=300, n_test=100, n_channels=3, contamination=0.05,
            random_state=42)[0]
        clf = MatrixProfile(window_size=20)
        clf.fit(X_multi)
        assert len(clf.decision_scores_) == 300

    def test_transductive_no_decision_function(self):
        clf = MatrixProfile(window_size=20)
        clf.fit(self.X_train)
        with self.assertRaises(NotImplementedError):
            clf.decision_function(np.random.randn(100))

    def test_transductive_no_predict(self):
        clf = MatrixProfile(window_size=20)
        clf.fit(self.X_train)
        with self.assertRaises(NotImplementedError):
            clf.predict(np.random.randn(100))

    def test_short_series_raises(self):
        with self.assertRaises(ValueError):
            clf = MatrixProfile(window_size=50)
            clf.fit(np.random.randn(30))

    def test_anomaly_scores_nonnegative(self):
        clf = MatrixProfile(window_size=20)
        clf.fit(self.X_train)
        # After threshold fill, some boundary scores equal threshold
        # All valid MP scores should be >= 0
        assert np.all(clf.decision_scores_ >= 0)


    def test_constant_channel_has_finite_results(self):
        x = np.random.default_rng(42).normal(size=80)
        X = np.column_stack([x, np.zeros_like(x)])
        for aggregation in ('max', 'mean'):
            with self.subTest(aggregation=aggregation):
                model = MatrixProfile(window_size=8,
                                      channel_aggregation=aggregation).fit(X)
                self.assertTrue(np.isfinite(model.decision_scores_).all())
                self.assertTrue(np.isfinite(model.threshold_))
                self.assertGreater(model.labels_.sum(), 0)

    def test_plateaus_do_not_poison_threshold(self):
        x = np.random.default_rng(42).normal(size=80)
        plateau = x.copy()
        plateau[:20] = 0
        for aggregation in ('max', 'mean'):
            for X in (plateau, np.column_stack([x, plateau])):
                with self.subTest(aggregation=aggregation, ndim=X.ndim):
                    model = MatrixProfile(window_size=8,
                                          channel_aggregation=aggregation).fit(X)
                    self.assertTrue(np.isfinite(model.decision_scores_).all())
                    self.assertTrue(np.isfinite(model.threshold_))

    def test_default_window_plateau(self):
        x = np.random.default_rng(43).normal(size=300)
        plateau = x.copy()
        plateau[:100] = 0
        for aggregation in ('max', 'mean'):
            for X in (plateau, np.column_stack([x, plateau])):
                with self.subTest(aggregation=aggregation, ndim=X.ndim):
                    model = MatrixProfile(
                        channel_aggregation=aggregation).fit(X)
                    self.assertTrue(np.isfinite(model.decision_scores_).all())
                    self.assertTrue(np.isfinite(model.threshold_))

    def test_all_constant_input_rejected(self):
        x = np.zeros(80)
        for X in (x, np.column_stack([x, x])):
            with self.subTest(ndim=X.ndim):
                with self.assertRaisesRegex(ValueError, 'no finite'):
                    MatrixProfile(window_size=8).fit(X)

    def test_nonfinite_window_scores_ignored_for_mean_and_max(self):
        from pyod.models._ts_utils import map_scores_to_timestamps
        windows = np.array([np.inf, 1.0, np.nan, 3.0])
        for aggregation in ('mean', 'max'):
            with self.subTest(aggregation=aggregation):
                scores, valid = map_scores_to_timestamps(
                    windows, window_size=2, step=1, n_timestamps=5,
                    aggregation=aggregation)
                np.testing.assert_array_equal(
                    valid, [False, True, True, True, True])
                np.testing.assert_allclose(scores[valid], [1, 1, 3, 3])


if __name__ == '__main__':
    unittest.main()
