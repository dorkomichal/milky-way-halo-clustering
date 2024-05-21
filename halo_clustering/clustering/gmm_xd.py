from extreme_deconvolution import extreme_deconvolution
from multiprocessing.pool import Pool
import numpy as np
from .parallel import get_max_processes
from .score import bayesian_information_criterion
from tqdm import tqdm
import os


def construct_covar_matrices(uncertainties: np.ndarray) -> np.ndarray:
    n_samples = uncertainties.shape[0]
    n_features = uncertainties.shape[1]
    covariances = np.empty((n_samples, n_features, n_features))
    it = 0
    for errs in uncertainties:
        covar = np.zeros((n_features, n_features))
        np.fill_diagonal(covar, errs**2)
        covariances[it] = covar
        it += 1
    return covariances


def generate_initial_guesses(
    num_of_components: int,
    features_min: np.ndarray,
    features_max: np.ndarray,
    num_features: int,
) -> tuple:
    xamp = np.ones(num_of_components) / 2.0
    xmean = np.array(
        [
            [
                np.random.uniform(features_min[i], features_max[i])
                for i in range(0, num_features)
            ]
            for _ in range(0, num_of_components)
        ]
    )
    xcovar = np.array(
        [np.diag(np.ones(num_features)) for _ in range(0, num_of_components)]
    )
    return xamp, xmean, xcovar


def run_xd(features: np.ndarray, uncertainties: np.ndarray) -> list:
    features_max = np.max(features, axis=0)
    features_min = np.min(features, axis=0)
    err_covar = construct_covar_matrices(uncertainties)
    bics_agg = list()
    fitted_params_agg = list()
    max_components = 10  # we are attempting to fit max 10 components
    sample_number = features.shape[0]
    num_features = features.shape[1]
    print(f"Running XD\n")
    for components in range(1, max_components + 1):
        bics = list()
        fitted_params = list()
        print(f"Attempting to fit {components} components\n")
        for _ in tqdm(range(0, 100)):
            xamp, xmean, xcovar = generate_initial_guesses(
                components, features_min, features_max, num_features
            )
            likelihood = extreme_deconvolution(features, err_covar, xamp, xmean, xcovar)
            bics.append(
                (
                    bayesian_information_criterion(
                        likelihood, components, num_features, sample_number
                    ),
                    components,
                )
            )
            fitted_params.append((xamp, xmean, xcovar))
        bics_agg.append(bics)
        fitted_params_agg.append(fitted_params)
    print("XD fitting complete")
    return bics_agg, fitted_params_agg


def xd_single_component(
    features: np.ndarray, uncertainties: np.ndarray, n_components: int
) -> list:
    features_max = np.max(features, axis=0)
    features_min = np.min(features, axis=0)
    err_covar = construct_covar_matrices(uncertainties)
    sample_number = features.shape[0]
    num_features = features.shape[1]
    print(f"Running XD\n")
    bics = list()
    fitted_params = list()
    print(f"Attempting to fit {n_components} components... PID: {os.getpid()}\n")
    number_of_iterations = 100
    for i in range(0, number_of_iterations):
        ## TQDM doesn't render nicely across multiple processes so use old style print statements to signify progress
        if i % 5 == 0:
            print(
                f"Fitting {n_components} components. Iteration number {i} out of {number_of_iterations}"
            )
        xamp, xmean, xcovar = generate_initial_guesses(
            n_components, features_min, features_max, num_features
        )
        likelihood = extreme_deconvolution(features, err_covar, xamp, xmean, xcovar)
        bics.append(
            (
                bayesian_information_criterion(
                    likelihood, n_components, num_features, sample_number
                ),
                n_components,
            )
        )
        fitted_params.append((xamp, xmean, xcovar))
    print(f"Fitting of {n_components} components finished\n")
    return bics, fitted_params


def run_xd_multiprocess(features: np.ndarray, uncertainties: np.ndarray) -> list:
    bics = list()
    fitted_params = list()
    print(f"Running XD fits in separate processes. One process per component fit\n")
    max_components = 10
    processes = get_max_processes()
    print(f"Spawning {processes} processes\n")
    with Pool(processes=processes) as process_pool:
        async_results = [
            process_pool.apply_async(
                xd_single_component, args=(features, uncertainties, n_cmp)
            )
            for n_cmp in range(1, max_components + 1)
        ]
        process_pool.close()
        process_pool.join()

        print(f"Fitting completed - all processes terminated. Collecting results.\n")
        for res in async_results:
            bics_res, fitted_params_res = res.get()
            bics.append(bics_res)
            fitted_params.append(fitted_params_res)
    return bics, fitted_params
