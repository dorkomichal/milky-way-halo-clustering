import math
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture


def bayesian_information_criterion(likelihood, n_components, n_features, n_samples):
    cov_params = n_components * n_features * (n_features + 1) / 2
    mean_params = n_features * n_components
    n_parameters = int(cov_params + mean_params + n_components - 1)
    return -2 * likelihood * n_samples + n_parameters * math.log(n_samples)


def process_bic_stats(bics: list) -> tuple:
    bics_stats = np.array(bics)
    components = bics_stats.shape[0]
    bic_min = np.zeros((components, 2))
    bic_median = np.zeros((components, 2))
    bic_max = np.zeros((components, 2))
    arg_min = np.zeros(components)
    for i in range(0, components):
        bic_max[i] = np.max(bics_stats[i], axis=0)
        bic_min[i] = np.min(bics_stats[i], axis=0)
        bic_median[i] = np.median(bics_stats[i], axis=0)
        arg_min[i] = np.argmin(bics_stats[i], axis=0)[0]

    return (bic_min, bic_max, bic_median, arg_min)


def best_fit_num_components_idx(bic_min: np.ndarray) -> int:
    return int(np.argmin(bic_min, axis=0)[0])


def pickle_bic_stats(bics: list, dataset_name: str) -> None:
    with open(f"./output/bics_raw_{dataset_name}.pickle", "wb") as pickle_file:
        pickle.dump(bics, pickle_file)


def pickle_fitted_params(params: list, dataset_name: str) -> None:
    with open(f"./output/fitted_params_{dataset_name}.pickle", "wb") as pickle_file:
        pickle.dump(params, pickle_file)


# Inspired by https://github.com/tholoien/XDGMM/blob/master/xdgmm/xdgmm.py#L172
def sklearn_gmm_cluster_membership(features, covar, n_components, xamp, xmean, xcovar):
    skl_gmm = GaussianMixture(n_components, covariance_type="full")
    skl_gmm.weights_ = xamp
    skl_gmm.means_ = xmean

    features = features[:, None, :]
    covar = covar[:, None, :, :]
    T = covar + xcovar

    probabilities = []

    for i in range(features.shape[0]):
        skl_gmm.covariances_ = T[i]
        skl_gmm.precisions_ = np.linalg.inv(T[i])
        skl_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(T[i]))
        probability = skl_gmm.predict_proba(features[i].reshape(1, -1))
        probabilities.append(probability)

    probabilities = np.array(probabilities)
    membership = np.argmax(probabilities, axis=2)
    return membership
