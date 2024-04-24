import math
import numpy as np
import pickle


def bayesian_information_criterion(likelihood, n_componets, n_features, n_samples):
    cov_params = n_componets * n_features * (n_features + 1) / 2
    mean_params = n_features * n_componets
    n_parameters = int(cov_params + mean_params + n_componets - 1)
    return n_parameters * math.log(n_samples) - 2 * likelihood * n_samples


def extract_bic_stats(bics: list) -> tuple:
    bics_stats = np.array(bics)
    components = bics_stats.shape[0]
    bic_min = np.zeros((components, 2))
    bic_median = np.zeros((components, 2))
    bic_max = np.zeros((components, 2))
    for i in range(0, components):
        bic_max[i] = np.max(bics_stats[i], axis=0)
        bic_min[i] = np.min(bics_stats[i], axis=0)
        bic_median[i] = np.median(bics_stats[i], axis=0)

    return (bic_min, bic_max, bic_median)


def pickle_bic_stats(bics: list, dataset_name: str) -> None:
    with open(f"./output/bics_raw_{dataset_name}.pickle", "wb") as pickle_file:
        pickle.dump(bics, pickle_file)
