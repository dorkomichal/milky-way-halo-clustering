from halo_clustering.clustering.score import (
    bayesian_information_criterion,
    best_fit_num_components_idx,
    process_bic_stats,
)
import math
import numpy as np


def test_best_fix_idx():
    bics_test = np.array(
        [
            [-1],  # 1 component
            [-2],  # 2 components
            [-4],  # 3 components
            [-7],  # 4 components
            [-5],  # 5 components
        ]
    )

    bic_min_idx = best_fit_num_components_idx(bics_test)
    assert bic_min_idx == 3


def test_bayesian_information_criterion():
    # Ok this is bit fabricated test
    likelihood = 7.412
    n_components = 7
    n_features = 6
    n_samples = 1200
    bic = bayesian_information_criterion(
        likelihood, n_components, n_features, n_samples
    )
    assert math.isclose(-16406.23501702366, bic)


def test_process_bic_stats():
    bics_stats = [
        [(-1, 1), (-1.5, 1), (-1.6, 1)],  # 1 component
        [(-4, 2), (-4.5, 2), (-3.9, 2)],  # 2 components
        [(-7, 3), (-8, 3), (-7.7, 3)],  # 3 components
        [(-5, 4), (-6.2, 4), (-6.8, 4)],  # 4 components
    ]

    bic_min, bic_max, bic_median, arg_min = process_bic_stats(bics_stats)
    bic_min_expected = np.array([[-1.6, 1], [-4.5, 2], [-8, 3], [-6.8, 4]])
    bic_max_expected = np.array([[-1, 1], [-3.9, 2], [-7, 3], [-5, 4]])
    bic_median_expected = np.array([[-1.5, 1], [-4, 2], [-7.7, 3], [-6.2, 4]])
    arg_min_expected = np.array([2, 1, 1, 2])

    assert np.allclose(bic_min, bic_min_expected)
    assert np.allclose(bic_max, bic_max_expected)
    assert np.allclose(bic_median, bic_median_expected)
    assert np.allclose(arg_min, arg_min_expected)
