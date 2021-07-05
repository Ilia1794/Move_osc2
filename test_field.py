import numpy as np
import os
import shutil
from Plot_funcs import plot_two_lines


def main():
    N = 1000
    data_first = np.zeros(N)
    data_second = np.zeros(N)
    x = np.linspace(1, 100, N)
    data_first = np.sin(data_first[0])
    data_second = np.sin(data_first[0])

    plot_two_lines(x, data_first, data_second)


def initial_fold(fold_name: str = 'Plots') -> None:
    if not os.path.exists(fold_name):
        os.mkdir(fold_name)
    else:
        shutil.rmtree(fold_name)
        os.mkdir(fold_name)


if __name__ == '__main__':
    initial_fold('Plots')
    main()