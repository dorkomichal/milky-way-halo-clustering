from argparse import ArgumentParser
import pandas as pd
from .clustering import preprocessing
from .clustering import gmm_xd
from .clustering.score import extract_bic_stats
from .clustering.visualisation import visualise_bic
import os


def xd_and_visualise(features, errors) -> None:
    bics = gmm_xd.run_xd(features, errors)
    bic_min, bic_max, bic_median = extract_bic_stats(bics)
    visualise_bic(bic_min, bic_max, bic_median)


def main(galah_filename: str, apogee_filename: str) -> None:
    print(f"Loading Galah dataset from {galah_filename}\n")
    galah_df = pd.read_csv(galah_filename)
    print(f"Loading Apogee dataset from {apogee_filename}\n")
    apogee_df = pd.read_csv(apogee_filename)
    apogee_features, apogee_errors = preprocessing.apogee_preprocess(apogee_df)
    galah_features, galah_errors = preprocessing.galah_preprocess(galah_df)
    print("Fitting GMM to Apogee dataset")
    xd_and_visualise(apogee_features, apogee_errors)
    # print("Fitting GMM to Apogee dataset")
    # xd_and_visualise(galah_features, galah_errors)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="MilkyWay Halo Clustering",
        description="Applies GMM to halo dataset from Apogee and Galah",
    )
    parser.add_argument("--galah", required=True)
    parser.add_argument("--apogee", required=True)
    args = parser.parse_args()
    ## Ensure output directory for plots exists
    if not os.path.exists("./output/"):
        os.makedirs("./output/")
    main(args.galah, args.apogee)
