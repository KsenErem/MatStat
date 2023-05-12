import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import statistics
from tabulate import tabulate
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

def multivariate_normal(size, ro):
    return stats.multivariate_normal.rvs([0, 0], [[1.0, ro], [ro, 1.0]], size=size)

def mix_multivariate_normal(size, ro):
    return 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size) + 0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)

def quadrant_coefficient(x, y):
    size = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    n = {1: 0, 2: 0, 3: 0, 4: 0}
    for i in range(size):
        if x[i] >= med_x and y[i] >= med_y:
            n[1] += 1
        elif x[i] < med_x and y[i] >= med_y:
            n[2] += 1
        elif x[i] < med_x and y[i] < med_y:
            n[3] += 1
        elif x[i] >= med_x and y[i] < med_y:
            n[4] += 1
    return (n[1] + n[3] - n[2] - n[4]) / size

def count_coefficients(get_sample, size, ro, repeats):
    pearson, quadrant, spirman = [], [], []
    for i in range(repeats):
        sample = get_sample(size, ro)
        x, y = sample[:, 0], sample[:, 1]
        pearson.append(stats.pearsonr(x, y)[0])
        spirman.append(stats.spearmanr(x, y)[0])
        quadrant.append(quadrant_coefficient(x, y))
    return pearson, spirman, quadrant

def create_table(pearson, spirman, quadrant, size, ro, repeats):
    if ro != -1:
        rows = [["rho = " + str(ro), 'r', 'r_{S}', 'r_{Q}']]
    else:
        rows = [["size = " + str(size), 'r', 'r_{S}', 'r_{Q}']]
    p = np.median(pearson)
    s = np.median(spirman)
    q = np.median(quadrant)
    rows.append(['E(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = np.median([pearson[k] ** 2 for k in range(repeats)])
    s = np.median([spirman[k] ** 2 for k in range(repeats)])
    q = np.median([quadrant[k] ** 2 for k in range(repeats)])
    rows.append(['E(z^2)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = statistics.variance(pearson)
    s = statistics.variance(spirman)
    q = statistics.variance(quadrant)
    rows.append(['D(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    return tabulate(rows, [], tablefmt="latex")

def build_ellipse(x, y, ax, n_std=3.0, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)

    print('\n'+'rad_x = '+str(rad_x))
    print('\n' + 'rad_y = ' + str(rad_y))

    ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor='none', **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def show_ellipse(size, ros):
    fig, ax = plt.subplots(1, 3)
    size_str = "n = " + str(size)
    titles = [size_str + r', $ \rho = 0$', size_str + r', $\rho = 0.5 $', size_str + r', $ \rho = 0.9$']
    for i in range(len(ros)):
        num, ro = i, ros[i]
        sample = multivariate_normal(size, ro)
        x, y = sample[:, 0], sample[:, 1]
        build_ellipse(x, y, ax[num], edgecolor='navy')
        ax[num].grid()
        ax[num].scatter(x, y, s=5)
        ax[num].set_title(titles[num])
    plt.show()