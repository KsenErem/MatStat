from CauchyDist.CauchyDist import plot_cauchy, print_table_cauchy, boxplot_Tukey_cauchy, draw_emp_func_cauchy, draw_kde_cauchy
from NormalDist.NormalDist import plot_normal, print_table_normal, boxplot_Tukey_normal, draw_emp_func_normal, draw_kde_normal
from PoissonDist.PoissonDist import plot_poisson, print_table_poisson, boxplot_Tukey_poisson, draw_emp_func_poisson, draw_kde_poisson
from UniformDist.UniformDist import plot_uniform, print_table_uniform, boxplot_Tukey_uniform, draw_emp_func_uniform, draw_kde_uniform
from LaplaceDist.LaplaceDist import plot_laplace
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sizes_1 = [10, 50, 1000]
    sizes_2 = [10, 100, 1000]
    sizes_3 = [20, 100]
    sizes_4 = [20, 60, 100]
    times = 1000
    koefs = [0.5, 1, 2]
    common_l, common_r = -4, 4
    poisson_l, poisson_r = 6, 14
    M = 10.0
    begin = -1.7320508
    length = begin * 2

    #Гистограммы и графики плотности распределения
    #plot_normal(sizes_1)
    #plot_cauchy(sizes_1, "cauchy nums", "denstiny", "cyan")
    #plot_poisson(M, sizes_1, "poisson nums", "denstiny", "cyan")
    #plot_uniform(begin, length, sizes_1, "uniform nums", "denstiny", "royalblue")
    #plot_laplace(sizes_1, "cauchy nums", "denstiny", "royalblue")

    #Характеристики положения и рассеяния
    #print_table_cauchy(sizes_2, times)
    #print_table_normal(sizes_2, times)
    #print_table_poisson(M, sizes_2, times)
    #print_table_uniform(sizes_2, times)

    #Боксплот Тьюки
    #boxplot_Tukey_normal(sizes_3, times)
    #boxplot_Tukey_cauchy(sizes_3, times)
    #boxplot_Tukey_poisson(M, sizes_3, times)
    #boxplot_Tukey_uniform(sizes_3, times)

    #Эмпирическая функция распределения
    #draw_emp_func_uniform(sizes_4, common_l, common_r)
    #draw_emp_func_normal(sizes_4, common_l, common_r)
    #draw_emp_func_cauchy(sizes_4, common_l, common_r)
    #draw_emp_func_poisson(M, sizes_4, poisson_l, poisson_r)

    #Ядерные оценки плотности распределения
    #draw_kde_normal(sizes_4, common_l, common_r, koefs)
    draw_kde_uniform(sizes_4, common_l, common_r, koefs)
    #draw_kde_cauchy(sizes_4, common_l, common_r, koefs)
    #draw_kde_poisson(M, sizes_4, poisson_l, poisson_r, koefs)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
