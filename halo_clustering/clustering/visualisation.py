import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import pandas as pd
from sklearn import manifold
from tabulate import tabulate
import umap


def visualise_bic(
    bic_min: np.ndarray, bic_max: np.ndarray, bic_median: np.ndarray, dataset_name: str
) -> None:
    """Visualise min, median and max BIC scores

    Saves figure into output file
    Args:
        bic_min (np.ndarray): min BIC for each component
        bic_max (np.ndarray): max BIC for each component
        bic_median (np.ndarray): median BIC for each component
        dataset_name (str): dataset name
    """
    plt.clf()
    plt.plot(bic_min.T[1], bic_min.T[0], label="Min BIC")
    plt.plot(bic_median.T[1], bic_median.T[0], linestyle="dashed", label="Median BIC")
    plt.plot(bic_max.T[1], bic_max.T[0], linestyle="dotted", label="Max BIC")
    plt.legend()
    plt.xticks([t for t in range(1, 11)])
    plt.xlabel("Number of Gaussian components")
    plt.ylabel("BIC")
    plt.savefig(f"./output/bic_figure_{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualise_bic_with_zoom(
    bic_min: np.ndarray, bic_max: np.ndarray, bic_median: np.ndarray, dataset_name: str
) -> None:
    """Visualise min, median and max BIC scores zooming around min score

    Saves figure into output file
    Args:
        bic_min (np.ndarray): min BIC for each component
        bic_max (np.ndarray): max BIC for each component
        bic_median (np.ndarray): median BIC for each component
        dataset_name (str): dataset name
    """
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
    plt.close()


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


def __xkcd_colours_select() -> list:
    """Get list of handpicked colours that are contrasting

    Returns:
        list: list of contrasting colours
    """
    return [
        "#fd3c06",  # red orange
        "#bf77f6",  # light purple
        "#0485d1",  # cerulean
        "#76cd26",  # apple green
        "#ff9408",  # tangerine
        "#b66a50",  # clay
        "#3d0734",  # aubergine
        "#fec615",  # golden yellow
        "#eecffe",  # pale levander
        "#978a84",  # warm grey
    ]


def __plot_features_by_cluster(
    feature_x: pd.Series,
    feature_y: pd.Series,
    cluster_membership: np.ndarray,
    x_label: str,
    y_label: str,
    num_of_components: int,
    dataset_name: str,
) -> None:
    """Plot x, y features and colour datapoints based on cluster membership

    Additionally draws confidence ellipse at 2-sigma
    Args:
        feature_x (pd.Series): feature on x-axis
        feature_y (pd.Series): feature on y-axis
        cluster_membership (np.ndarray): array of cluster memberships
        x_label (str): label for x-axis
        y_label (str): label for y-axis
        num_of_components (int): number of fitted components
        dataset_name (str): name of dataset fitted
    """
    plt.clf()
    cluster_max = np.max(cluster_membership)
    xkcd_colour_list = __xkcd_colours_select()
    cluster_colours = [xkcd_colour_list[i] for i in range(0, cluster_max + 1)]
    colour_map = [xkcd_colour_list[cluster[0]] for cluster in cluster_membership]
    _, ax = plt.subplots()
    ax.scatter(feature_x, feature_y, s=10, c=colour_map)
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
        f"./output/clusters_{dataset_name}_{num_of_components}components_{x_feature_filename}_{y_feature_filename}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def __apogee_visualisation_feature_pairs() -> list:
    """Get list of pairs of features and axis labels for apogee dataset

    Returns:
        list: pairs of features and axis labels
    """
    return [
        ("FE_H", "Fe/H", "ALPHA_FE", "\u03b1/Fe"),
        ("FE_H", "Fe/H", "E_SCALED", "scaled Energy"),
        ("AL_FE", "Al/Fe", "E_SCALED", "scaled Energy"),
        ("FE_H", "Fe/H", "CE_FE", "Ce/Fe"),
        ("FE_H", "Fe/H", "AL_FE", "Al/Fe"),
        ("AL_FE", "Al/Fe", "MG_MN", "Mg/Mn"),
    ]


def __galah_visualisation_feature_pairs() -> list:
    """Get list of pairs of features and axis labels for galah dataset

    Returns:
        list: pairs of features and axis labels
    """
    return [
        ("fe_h", "Fe/H", "alpha_fe", "\u03b1/Fe"),
        ("fe_h", "Fe/H", "scaled_Energy", "scaled Energy"),
        ("Al_fe", "Al/Fe", "scaled_Energy", "scaled Energy"),
        ("fe_h", "Fe/H", "Y_fe", "Y/Fe"),
        ("fe_h", "Fe/H", "Al_fe", "Al/Fe"),
        ("Al_fe", "Al/Fe", "Mg_mn", "Mg/Mn"),
        ("fe_h", "Fe/H", "Mn_fe", "Mn/Fe"),
        ("fe_h", "Fe/H", "Na_fe", "Na/Fe"),
        ("Na_fe", "Na/Fe", "Mg_cu", "Mg/Cu"),
        ("fe_h", "Fe/H", "Ba_fe", "Ba/Fe"),
        ("fe_h", "Fe/H", "Eu_fe", "Eu/Fe"),
        ("fe_h", "Fe/H", "Ba_eu", "Ba/Eu"),
    ]


def visualise_features(
    features: pd.DataFrame,
    cluster_membership: np.ndarray,
    num_of_components: int,
    dataset_name: str,
) -> None:
    """Visualise features as 2-dimensional chemodynamical projections

    Produce scatter plots and save them into the file
    Args:
        features (pd.DataFrame): features to visualise
        cluster_membership (np.ndarray): cluster membership of each point in the dataset
        num_of_components (int): number of components fitted into the dataset
        dataset_name (str): name of the dataset

    Raises:
        f: exception if provided unsupported dataset
    """
    if dataset_name == "apogee":
        visualisation_pairs = __apogee_visualisation_feature_pairs()
    elif dataset_name == "galah":
        visualisation_pairs = __galah_visualisation_feature_pairs()
    else:
        raise f"Unknown dataset {dataset_name}"

    for pair in visualisation_pairs:
        x_feature = features[pair[0]]
        x_display_label = pair[1]
        y_feature = features[pair[2]]
        y_display_label = pair[3]
        __plot_features_by_cluster(
            x_feature,
            y_feature,
            cluster_membership,
            x_display_label,
            y_display_label,
            num_of_components,
            dataset_name,
        )


def tabulate_components(
    xamp: np.ndarray,
    xmean: np.ndarray,
    xcovar: np.ndarray,
    number_components: int,
    num_samples: int,
    dataset_name: str,
) -> None:
    """Produce table listing all features (mean and sigma), count and weight by component

    Saves table into text file
    Args:
        xamp (np.ndarray): amplitudes fitted by XD
        xmean (np.ndarray): means fitted by XD
        xcovar (np.ndarray): covariances fitted by XD
        number_components (int): number of components fitted
        num_samples (int): number of samples
        dataset_name (str): name of the dataset
    """
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
            f"Component {i + 1}",
            f"{xamp[i] * 100: .2f} %",
            int(num_samples * xamp[i]),
        ]
        for j in range(0, num_features):
            table_means.append(f"{xmean[i,j]: .2f} +-{xcovar[i,j,j]**0.5: .2f}")
        table.append(table_means)

    output = tabulate(table, headers=headers_combined)
    print(output)
    with open(
        f"./output/components{number_components}_tabulate_{dataset_name}.txt", "w"
    ) as out_file:
        out_file.write(output)


def __plot_tsne_with_clusters(
    feature_x: pd.Series,
    feature_y: pd.Series,
    cluster_membership: np.ndarray,
    num_components: int,
    dataset_name: str,
    perplexity: int,
) -> None:
    """Plot t-SNE projection with data points coloured by cluster membership

    Save figure into output file
    Args:
        feature_x (pd.Series): embedding on x-axis
        feature_y (pd.Series): embedding on x-axis
        cluster_membership (np.ndarray): cluster membership
        num_components (int): number of components
        dataset_name (str): datase name
        perplexity (int): perplexity hyperparam used (for file name)
    """
    plt.clf()
    xkcd_colour_list = __xkcd_colours_select()
    colour_map = [xkcd_colour_list[cluster[0]] for cluster in cluster_membership]
    _, ax = plt.subplots()
    ax.scatter(feature_x, feature_y, s=10, c=colour_map)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.savefig(
        f"./output/TSNE_{dataset_name}_{num_components}components_perplexity_{perplexity}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def visualise_features_tsne(
    features_np: np.ndarray,
    cluster_membership: np.ndarray,
    num_components: int,
    dataset_name: str,
) -> None:
    """Compute t-SNE embedding for the provided features and plot them

    Saves figures into file.
    Applies perplexities [5, 30, 50, 100]
    Args:
        features_np (np.ndarray): features to compute t-SNE embedding for
        cluster_membership (np.ndarray): cluster membership of datapoints
        num_components (int): number of components fitted into the data
        dataset_name (str): name of the dataset
    """
    perplexities = [
        5,
        30,
        50,
        100,
    ]  # chosen as in https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html

    for perplexity in perplexities:
        tsne = manifold.TSNE(
            n_components=2,
            init="random",
            random_state=42,
            perplexity=perplexity,
            n_iter=300,
        )
        features_tsne = tsne.fit_transform(features_np)
        x_feature = features_tsne[:, 0]
        y_feature = features_tsne[:, 1]
        __plot_tsne_with_clusters(
            x_feature,
            y_feature,
            cluster_membership,
            num_components,
            dataset_name,
            perplexity,
        )


def __plot_umap_with_clusters(
    feature_x: pd.Series,
    feature_y: pd.Series,
    cluster_membership: np.ndarray,
    num_components: int,
    dataset_name: str,
) -> None:
    """Plot UMAP projection with data points coloured by cluster membership

    Save figure into output file
    Args:
        feature_x (pd.Series): embedding on x-axis
        feature_y (pd.Series): embedding on x-axis
        cluster_membership (np.ndarray): cluster membership
        num_components (int): number of components
        dataset_name (str): datase name
    """
    plt.clf()
    xkcd_colour_list = __xkcd_colours_select()
    colour_map = [xkcd_colour_list[cluster[0]] for cluster in cluster_membership]
    _, ax = plt.subplots()
    ax.scatter(feature_x, feature_y, s=10, c=colour_map)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.savefig(
        f"./output/UMAP_{dataset_name}_{num_components}_clusters.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def visualise_features_umap(
    features_np: np.ndarray,
    cluster_membership: np.ndarray,
    num_components: int,
    dataset_name: str,
) -> None:
    """Compute UMAP embedding for the provided features and plot them

    Saves figures into file.
    Args:
        features_np (np.ndarray): features to compute t-SNE embedding for
        cluster_membership (np.ndarray): cluster membership of datapoints
        num_components (int): number of components fitted into the data
        dataset_name (str): name of the dataset
    """
    umap_mapper = umap.UMAP(
        n_components=2,
        init="spectral",
        learning_rate=1.0,
        local_connectivity=1.0,
        low_memory=False,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    )
    features_umap = umap_mapper.fit_transform(features_np)
    x_feature = features_umap[:, 0]
    y_feature = features_umap[:, 1]
    __plot_umap_with_clusters(
        x_feature, y_feature, cluster_membership, num_components, dataset_name
    )
