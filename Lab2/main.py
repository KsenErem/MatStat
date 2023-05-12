import numpy as np
import scipy.stats as stats
import math as m
from task1 import count_coefficients, create_table, multivariate_normal, show_ellipse, mix_multivariate_normal
from task2 import reference_noisy_function, plot_linear_regression
from task3 import task3Solver
from task4 import normal, mean, dispersion_exp, draw_result, task4, task4_asymp

# TASK 1 CODE
def task1_builder():
    sizes = [20, 60, 100]
    ros = [0, 0.5, 0.9]
    REPETITIONS = 1000

    for size in sizes:
        for ro in ros:
            pearson, spirman, quadrant = count_coefficients(multivariate_normal, size, ro, REPETITIONS)
            print('\n' + str(size) + '\n' + str(create_table(pearson, spirman, quadrant, size, ro, REPETITIONS)))

        pearson, spearman, quadrant = count_coefficients(mix_multivariate_normal, size, 0, REPETITIONS)
        print('\n' + str(size) + '\n' + str(create_table(pearson, spirman, quadrant, size, -1, REPETITIONS)))
        show_ellipse(size, ros)
    return

#task1_builder()

# TASK 2 CODE
def task2_builder():
    # Without any perturbations
    x = np.arange(-1.8, 2, 0.2)
    y = reference_noisy_function(x)
    plot_linear_regression('NoPerturbations', x, y)

    # With perturbations in first and last elements
    x = np.arange(-1.8, 2, 0.2)
    y = reference_noisy_function(x)
    y[0] += 10
    y[-1] -= 10
    plot_linear_regression('Perturbations', x, y)
    return

task2_builder()

# TASK 3 CODE
def task3_builder():
    sizes = [20, 100]
    alpha = 0.05
    p = 1 - alpha

    # for normal ditribution
    task3Solver(sizes[1], np.random.normal(0, 1, size=sizes[1]), p, alpha)

    # for laplace ditribution
    task3Solver(sizes[0], stats.laplace.rvs(size=sizes[0], scale=1 / m.sqrt(2), loc=0), p, alpha)

    # for uniform ditribution
    task3Solver(sizes[0], stats.uniform.rvs(size=sizes[0], loc=-m.sqrt(3), scale=2 * m.sqrt(3)), p, alpha)

    return

#task3_builder()

# TASK 4 CODE
def task4_builder():
    n_set = [20, 100]
    x_20 = normal(20)
    x_100 = normal(100)
    x_set = [x_20, x_100]
    task4(x_set, n_set)
    task4_asymp(x_set, n_set)
    return

#task4_builder()