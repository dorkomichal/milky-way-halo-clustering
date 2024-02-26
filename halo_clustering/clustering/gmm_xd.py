import numpy as np
from .score import bayesian_information_criterion
from extreme_deconvolution import extreme_deconvolution
from tqdm import tqdm


def construct_covar_matrices(uncertainties: np.ndarray) -> np.ndarray:
    n_samples = uncertainties.shape[0]
    n_features = uncertainties.shape[1]
    covariances = np.empty((n_samples, n_features, n_features))
    it = 0
    for errs in uncertainties:
        covar = np.zeros((n_features, n_features))
        np.fill_diagonal(covar, errs)
        covariances[it] = covar
        it += 1
    return covariances


def generate_initial_guesses(
    num_of_components: int, features_min: np.ndarray, features_max: np.ndarray
) -> tuple:
    xamp = np.ones(num_of_components) / 2.0
    xmean = np.array(
        [
            [np.random.uniform(features_min[i], features_max[i]) for i in range(0, 6)]
            for _ in range(0, num_of_components)
        ]
    )
    xcovar = np.array([np.diag(np.ones(6)) for _ in range(0, num_of_components)])
    return xamp, xmean, xcovar


def run_xd(features: np.ndarray, uncertainties: np.ndarray):
    features_max = np.max(features, axis=0)
    features_min = np.min(features, axis=0)
    err_covar = construct_covar_matrices(uncertainties)
    bics_agg = list()
    max_components = 10  # we are attempting to fit max 10 components
    sample_number = features.shape[0]
    print(f"Running XD\n")
    for components in range(1, max_components):
        bics = list()
        print(f"Attempting to fit {components} components\n")
        for _ in tqdm(range(0, 10)):
            xamp, xmean, xcovar = generate_initial_guesses(
                components, features_min, features_max
            )
            likelihood = extreme_deconvolution(features, err_covar, xamp, xmean, xcovar)
            bics.append(
                (
                    bayesian_information_criterion(
                        likelihood, components, features.shape[1], sample_number
                    ),
                    components,
                )
            )
        bics_agg.append(bics)
    print("XD fitting complete")
    return bics_agg
