from argparse import ArgumentParser
from numpy.typing import NDArray
import pandas as pd
from .clustering import preprocessing
from .clustering import gmm_xd
from .clustering import parallel
from .clustering.score import (
    best_fit_num_components_idx,
    process_bic_stats,
    pickle_bic_stats,
    pickle_fitted_params,
    sklearn_gmm_cluster_membership,
)
from .clustering.visualisation import (
    tabulate_components,
    visualise_bic,
    visualise_bic_with_zoom,
    visualise_features,
    visualise_features_tsne,
    visualise_features_umap,
)
import os


def xd_and_visualise(
    features: pd.DataFrame, errors: pd.DataFrame, multiprocess: bool, dataset_name: str
) -> None:
    features_np = features.to_numpy()
    errors_np = errors.to_numpy()
    bics, fitted_params = (
        gmm_xd.run_xd_multiprocess(features_np, errors_np)
        if multiprocess
        else gmm_xd.run_xd(features_np, errors_np)
    )
    pickle_bic_stats(bics, dataset_name)
    pickle_fitted_params(fitted_params, dataset_name)
    bic_min, bic_max, bic_median, arg_min = process_bic_stats(bics)
    visualise_bic(bic_min, bic_max, bic_median, dataset_name)
    visualise_bic_with_zoom(bic_min, bic_max, bic_median, dataset_name)
    num_components_idx = best_fit_num_components_idx(bic_min)
    if dataset_name == "galah":
        # due to low data resolution resulting in best BIC for 3 components also visualise 4 and 5 components for Galah
        components_idx = [
            num_components_idx,
            num_components_idx + 1,
            num_components_idx + 2,
        ]
    else:
        # Apogee 2d projection for the favoured number of components and -1,+1,+2 components around the favoured
        components_idx = [
            num_components_idx - 1,
            num_components_idx,
            num_components_idx + 1,
            num_components_idx + 2,
        ]

    for component_idx in components_idx:
        num_components = component_idx + 1
        xamp, xmean, xcovar = fitted_params[component_idx][int(arg_min[component_idx])]
        features_covar = gmm_xd.construct_covar_matrices(errors_np)
        cluster_membership = sklearn_gmm_cluster_membership(
            features_np, features_covar, num_components, xamp, xmean, xcovar
        )
        visualise_features(features, cluster_membership, num_components, dataset_name)
        tabulate_components(
            xamp, xmean, xcovar, num_components, features_np.shape[0], dataset_name
        )
        visualise_features_tsne(
            features_np, cluster_membership, num_components, dataset_name
        )
        visualise_features_umap(
            features_np, cluster_membership, num_components, dataset_name
        )


def main(galah_filename: str, apogee_filename: str, multiprocess: bool) -> None:
    print(f"Loading Galah dataset from {galah_filename}\n")
    galah_df = pd.read_csv(galah_filename)
    print(f"Loading Apogee dataset from {apogee_filename}\n")
    apogee_df = pd.read_csv(apogee_filename)
    apogee_features, apogee_errors = preprocessing.apogee_preprocess(apogee_df)
    galah_features, galah_errors = preprocessing.galah_preprocess(galah_df)
    print("Fitting GMM to Apogee dataset")
    xd_and_visualise(apogee_features, apogee_errors, multiprocess, "apogee")
    print("Fitting GMM to Galah dataset")
    xd_and_visualise(galah_features, galah_errors, multiprocess, "galah")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="MilkyWay Halo Clustering",
        description="Applies GMM to halo dataset from Apogee and Galah",
    )
    parser.add_argument("--galah", required=True)
    parser.add_argument("--apogee", required=True)
    parser.add_argument("--multiprocess", action="store_const", const=True)
    args = parser.parse_args()
    ## Ensure output directory for plots exists
    if not os.path.exists("./output/"):
        os.makedirs("./output/")
    multiprocess = True if args.multiprocess else False
    if multiprocess:
        parallel.initialize_multiprocessing()
    else:
        os.environ["OMP_NUM_THREADS"] = f"{os.cpu_count()}"
    print(f"Multiprocess value {multiprocess}\n")
    main(args.galah, args.apogee, multiprocess)
