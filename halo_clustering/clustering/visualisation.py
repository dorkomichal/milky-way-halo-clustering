import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import pandas as pd
from tabulate import tabulate


def visualise_bic(bic_min, bic_max, bic_median, dataset_name) -> None:
    plt.clf()
    plt.plot(bic_min.T[1], bic_min.T[0], label="Min BIC")
    plt.plot(bic_median.T[1], bic_median.T[0], linestyle="dashed", label="Median BIC")
    plt.plot(bic_max.T[1], bic_max.T[0], linestyle="dotted", label="Max BIC")
    plt.legend()
    plt.xticks([t for t in range(1, 11)])
    plt.xlabel("Number of Gaussian components")
    plt.ylabel("BIC")
    plt.savefig(f"./output/bic_figure_{dataset_name}.png", dpi=300, bbox_inches="tight")


def visualise_bic_with_zoom(bic_min, bic_max, bic_median, dataset_name) -> None:
    plt.clf()
    _, ax = plt.subplots()
    ax.plot(bic_min.T[1], bic_min.T[0], label="Min BIC")
    ax.plot(bic_median.T[1], bic_median.T[0], linestyle="dashed", label="Median BIC")
    ax.plot(bic_max.T[1], bic_max.T[0], linestyle="dotted", label="Max BIC")

    min_components = np.argmin(bic_min.T[0]) + 1
    min_bic_y = np.min(bic_min.T[0])
    x1_zoom = min_components - 1
    x2_zoom = min_components + 1
    y_min, y_max = ax.get_ylim()
    y1_zoom = min_bic_y
    y2_zoom = min_bic_y + (y_max - y_min) / 10

    # Make the zoom-in plot:
    axins = zoomed_inset_axes(ax, 1, loc="upper center")  # zoom = 1
    axins.plot(bic_min.T[1], bic_min.T[0])
    axins.plot(bic_median.T[1], bic_median.T[0], linestyle="dashed")
    axins.plot(bic_max.T[1], bic_max.T[0], linestyle="dotted")
    axins.set_xlim(x1_zoom - 0.5, x2_zoom + 0.5)
    axins.set_ylim(y1_zoom, y2_zoom)
    axins.set_xticks([t for t in range(x1_zoom, x2_zoom + 1)])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.7")
    ax.set_xlabel("Number of Gaussian components")
    ax.set_xticks([t for t in range(1, 11)])
    ax.set_ylabel("BIC")
    ax.legend()
    plt.savefig(
        f"./output/bic_figure_zoom_{dataset_name}.png", dpi=300, bbox_inches="tight"
    )


# Ref: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def __plot_features_by_cluster(
    feature_x: pd.Series,
    feature_y: pd.Series,
    cluster_membership: np.ndarray,
    x_label: str,
    y_label: str,
    dataset_name: str,
) -> None:
    cluster_max = np.max(cluster_membership)
    norm = mpl.colors.Normalize(vmin=0, vmax=cluster_max)
    cluster_colours = [cm.viridis(norm(i)) for i in range(0, cluster_max + 1)]
    _, ax = plt.subplots()
    ax.scatter(feature_x, feature_y, s=10, c=cluster_membership, norm=norm)
    for cluster in range(0, cluster_max + 1):
        indices = np.where(cluster_membership == cluster)[0]
        feature_x_cluster = feature_x.iloc[indices]
        feature_y_cluster = feature_y.iloc[indices]
        confidence_ellipse(
            feature_x_cluster,
            feature_y_cluster,
            ax,
            n_std=2,
            edgecolor=cluster_colours[cluster],
        )
    ax.set_xlabel(f"[{x_label}]")
    ax.set_ylabel(f"[{y_label}]")
    x_feature_filename = x_label.replace("/", "")
    y_feature_filename = y_label.replace("/", "")
    plt.savefig(
        f"./output/clusters_{dataset_name}_{x_feature_filename}_{y_feature_filename}.png",
        dpi=300,
        bbox_inches="tight",
    )


def visualise_features(
    features: pd.DataFrame, cluster_membership: np.ndarray, dataset_name: str
):
    if dataset_name == "apogee":
        fe_h_apogee = features["FE_H"]
        al_fe_apogee = features["AL_FE"]
        __plot_features_by_cluster(
            fe_h_apogee, al_fe_apogee, cluster_membership, "Fe/H", "Al/Fe", dataset_name
        )
        # TODO plot remaining features
    elif dataset_name == "galah":
        fe_h_galah = features["fe_h"]
        alpha_fe_galah = features["alpha_fe"]
        __plot_features_by_cluster(
            fe_h_galah,
            alpha_fe_galah,
            cluster_membership,
            "Fe/H",
            "\u03b1/Fe",
            dataset_name,
        )
        # TODO plot remaining features
    else:
        raise f"Unknown dataset {dataset_name}"


def tabulate_components(
    xamp: np.ndarray,
    xmean: np.ndarray,
    xcovar: np.ndarray,
    number_components: int,
    num_samples: int,
    dataset_name: str,
) -> None:
    apogee_feature_cols = [
        "Energy",
        "[Fe/H]",
        "[Alpha/Fe]",
        "[Al_Fe]",
        "[Ce/Fe]",
        "[Mg/Mn]",
    ]

    galah_feature_cols = [
        "Energy",
        "[Fe/H]",
        "[Alpha/Fe]",
        "[Na/Fe]",
        "[Al/Fe]",
        "[Mn/Fe]",
        "[Y/Fe]",
        "[Ba/Fe]",
        "[Eu/Fe]",
        "[Mg/Cu]",
        "[Mg/Mn]",
        "[Ba/Eu]",
    ]

    headers = ["Component", "Weight", "Count"]
    headers_combined = (
        headers + apogee_feature_cols
        if dataset_name == "apogee"
        else headers + galah_feature_cols
    )

    num_features = (
        len(apogee_feature_cols)
        if dataset_name == "apogee"
        else len(galah_feature_cols)
    )
    table = []
    for i in range(0, number_components):
        table_means = [
            f"Component {i}",
            f"{xamp[i] * 100: .2f} %",
            int(num_samples * xamp[i]),
        ]
        for j in range(0, num_features):
            table_means.append(f"{xmean[i,j]: .2f} +-{xcovar[i,j,j]**0.5: .2f}")
        table.append(table_means)

    output = tabulate(table, headers=headers_combined)
    print(output)
    with open(f"./output/components_tabulate_{dataset_name}.txt", "w") as out_file:
        out_file.write(output)
