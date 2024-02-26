import matplotlib.pyplot as plt


def visualise_bic(bic_min, bic_max, bic_median) -> None:
    plt.plot(bic_min.T[1], bic_min.T[0], label="Min BIC")
    plt.plot(bic_median.T[1], bic_median.T[0], linestyle="dashed", label="Median BIC")
    plt.plot(bic_max.T[1], bic_max.T[0], linestyle="dotted", label="Max BIC")
    plt.legend()
    plt.xticks([t for t in range(1, 11)])
    plt.xlabel("Number of Gaussian components")
    plt.ylabel("BIC")
    plt.savefig("./output/bic_figure.png", dpi=300, bbox_inches="tight")
