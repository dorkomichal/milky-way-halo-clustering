import pandas as pd


def apogee_preprocess(apogee_df: pd.DataFrame):
    feature_cols = [
        "E_SCALED",
        "FE_H",
        "ALPHA_FE",
        "AL_FE",
        "CE_FE",
        "MG_MN",
    ]
    uncertainties_cols = [
        "E_SCALED_ERR",
        "FE_H_ERR",
        "ALPHA_FE_ERR",
        "AL_FE_ERR",
        "CE_FE_ERR",
        "MG_MN_ERR",
    ]

    features = apogee_df[feature_cols]
    uncertainties = apogee_df[uncertainties_cols]
    return (features, uncertainties)


def galah_preprocess(galah_df: pd.DataFrame):
    features_cols = [
        "scaled_Energy",
        "fe_h",
        "alpha_fe",
        "Na_fe",
        "Al_fe",
        "Mn_fe",
        "Y_fe",
        "Ba_fe",
        "Eu_fe",
        "Mg_cu",
        "Mn_fe",
        "Ba_fe",
    ]

    uncertainties_cols = [
        "scaled_e_Energy",
        "e_fe_h",
        "e_alpha_fe",
        "e_Na_fe",
        "e_Al_fe",
        "e_Mn_fe",
        "e_Y_fe",
        "e_Ba_fe",
        "e_Eu_fe",
        "e_Mg_cu",
        "e_Mn_fe",
        "e_Ba_fe",
    ]

    features = galah_df[features_cols]
    uncertainties = galah_df[uncertainties_cols]
    return (features, uncertainties)
