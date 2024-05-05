import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np


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
