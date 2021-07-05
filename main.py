import os
import shutil

import pyximport

pyximport.install()

# TEST CHANGE

from cython_module import hello_world
from Plot_funcs import plot_two_lines
from volterra import volterra_compute

def main():
    hello_world()
    abscissa, data_first, data_second = volterra_compute()
    plot_two_lines(abscissa, data_first, data_second)


def initial_fold(fold_name: str = 'Plots') -> None:
    if not os.path.exists(fold_name):
        os.mkdir(fold_name)
    else:
        shutil.rmtree(fold_name)
        os.mkdir(fold_name)


if __name__ == '__main__':
    initial_fold('Plots')
    main()
