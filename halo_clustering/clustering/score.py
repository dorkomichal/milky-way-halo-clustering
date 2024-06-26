import math
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture


def bayesian_information_criterion(
    likelihood: float, n_components: int, n_features: int, n_samples: int
) -> float:
    """Compute BIC score

    Args:
        likelihood (float): likelihood
        n_components (int): number of components fitted
        n_features (int): number of features in the dataset
        n_samples (int): number of samples in the dataset

    Returns:
        float: bic score
    """
    cov_params = n_components * n_features * (n_features + 1) / 2
    mean_params = n_features * n_components
    n_parameters = int(cov_params + mean_params + n_components - 1)
    return -2 * likelihood * n_samples + n_parameters * math.log(n_samples)


def process_bic_stats(
    bics: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process accumulated BIC stats from XD fitting and return min, max, median, argmin

    Args:
        bics (list): Accumulated BIC stats

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: BIC min, max, median and argmin
    """
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
    """Get index of element corresponding to the lowest BIC score

    Args:
        bic_min (np.ndarray): minimum BIC scores for each component fit

    Returns:
        int: index of lowest BIC score
    """
    return int(np.argmin(bic_min, axis=0)[0])


def pickle_bic_stats(bics: list, dataset_name: str) -> None:
    """Store raw BIC stats in the pickle file

    Args:
        bics (list): accumulated BIC stats
        dataset_name (str): name of dataset
    """
    with open(f"./output/bics_raw_{dataset_name}.pickle", "wb") as pickle_file:
        pickle.dump(bics, pickle_file)


def pickle_fitted_params(params: list, dataset_name: str) -> None:
    """Pickle raw fitted params

    Args:
        params (list): fitted raw params
        dataset_name (str): dataset name
    """
    with open(f"./output/fitted_params_{dataset_name}.pickle", "wb") as pickle_file:
        pickle.dump(params, pickle_file)


# Inspired by https://github.com/tholoien/XDGMM/blob/master/xdgmm/xdgmm.py#L172
def sklearn_gmm_cluster_membership(
    features: np.ndarray,
    covar: np.ndarray,
    n_components: int,
    xamp: np.ndarray,
    xmean: np.ndarray,
    xcovar: np.ndarray,
) -> np.ndarray:
    """Compute cluster membership using scikit-learn GaussianMixture class

    Args:
        features (np.ndarray): features fitted by XD
        covar (np.ndarray): errors of measurements of features
        n_components (int): number of components fitted
        xamp (np.ndarray): amplitude of gaussians fitted by XD
        xmean (np.ndarray): means of gaussians fitted by XD
        xcovar (np.ndarray): covariance of gaussians fitted by XD

    Returns:
        np.ndarray: cluster membership of each datapoint in the sample
    """
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
