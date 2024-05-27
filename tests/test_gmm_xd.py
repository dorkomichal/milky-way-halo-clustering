from halo_clustering.clustering.gmm_xd import (
    construct_covar_matrices,
    generate_initial_guesses,
)
import math
import numpy as np


def test_construct_covar_matrices():
    test_uncertainties = np.array(
        [
            # 3 features
            [0.2, 0.4, 0.35],  # sample 1
            [0.1, 0.2, 0.33],  # sample 2
            [0.2, 0.4, 0.35],  # sample 3
            [0.1, 0.2, 0.33],  # sample 4
            [0.2, 0.4, 0.35],  # sample 5
        ]
    )

    expected_covar = np.array(
        [
            [[0.04, 0, 0], [0, 0.16, 0], [0, 0, 0.1225]],  # sample 1
            [[0.01, 0, 0], [0, 0.04, 0], [0, 0, 0.1089]],  # sample 2
            [[0.04, 0, 0], [0, 0.16, 0], [0, 0, 0.1225]],  # sample 3
            [[0.01, 0, 0], [0, 0.04, 0], [0, 0, 0.1089]],  # sample 4
            [[0.04, 0, 0], [0, 0.16, 0], [0, 0, 0.1225]],  # sample 5
        ]
    )

    covar = construct_covar_matrices(test_uncertainties)
    assert np.allclose(covar, expected_covar)


def test_generate_initial_guesses():
    # def generate_initial_guesses
    # num_of_components: int,
    # features: np.ndarray,
    # num_features: int,
    num_of_components = 4
    num_features = 3
    test_features = np.array(
        [
            # 3 features
            [1.43, 0.4, 1.35],  # sample 1
            [1.14, 0.2, 2.33],  # sample 2
            [1.76, 0.4, 3.22],  # sample 3
            [1.13, 0.2, 3.12],  # sample 4
            [1.29, 0.4, 1.47],  # sample 5
        ]
    )

    xamp, xmean, xcovar = generate_initial_guesses(
        num_of_components, test_features, num_features
    )

    expected_covar = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert math.isclose(1.0, np.sum(xamp))  # amplitudes should sum up to 1
    for i in range(num_features):
        assert np.allclose(xcovar[i], expected_covar)

    for i in range(num_of_components):
        assert xmean[i, 0] in test_features[:, 0]
        assert xmean[i, 1] in test_features[:, 1]
        assert xmean[i, 2] in test_features[:, 2]
