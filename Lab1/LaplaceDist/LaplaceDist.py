import math as m
from scipy.stats import laplace
import matplotlib.pyplot as plt
import numpy as np

def plot_laplace(sizes: list, x_label: str, y_label: str, clr: str):
    for size in sizes:
        density = laplace(scale=1 / m.sqrt(2), loc=0)
        histogram = laplace.rvs(size=size, scale=1 / m.sqrt(2), loc=0)
        fig, ax = plt.subplots(1, 1)
        ax.hist(histogram, density=True, histtype='stepfilled', alpha=0.6, color=clr)
        x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
        ax.plot(x, density.pdf(x), 'k--', lw=1.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("n = " + str(size))
        plt.grid()
        #plt.show()
        fig.savefig(".//LaplaceDist" + "laplace" + str(size) + ".png")
    return
