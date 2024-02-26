import matplotlib.pyplot as plt


def visualise_bic(bic_min, bic_max, bic_median) -> None:
    plt.plot(bic_min.T[1], bic_min.T[0])
    plt.plot(bic_median.T[1], bic_median.T[0], linestyle="dashed")
    plt.plot(bic_max.T[1], bic_max.T[0], linestyle="dotted")
    plt.gca().invert_yaxis()
    plt.xlabel("Number of Gaussian components")
    plt.ylabel("BIC")
    plt.tight_layout()
    plt.savefig("./output/bic_figure.png")
