import os
import shutil
import numpy as np
import pyximport

pyximport.install()

from cython_module import hello_world
from Plot_funcs import plot_two_lines
from volterra import volterra_compute


def main():
    abscissa, numeric, analytic, p, force = volterra_compute()
    plot_two_lines(abscissa, numeric, analytic, 'Force', 't', 'P', 'Numeric', 'Analytic')
    plot_two_lines(abscissa, p, force, 'Force_external', 't', 'Force', 'p', 'force')
    plot_two_lines(abscissa, np.abs(np.fft.fft(force)), np.angle(np.fft.fft(force)),
                   'Force_abs_angle', 't', 'Force', 'abs',
                   'angle')


def initial_fold(fold_name: str = 'Plots') -> None:
    if not os.path.exists(fold_name):
        os.mkdir(fold_name)
    else:
        shutil.rmtree(fold_name)
        os.mkdir(fold_name)


if __name__ == '__main__':
    initial_fold('Plots')
    main()
