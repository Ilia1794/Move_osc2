import os
import shutil
import numpy as np
import scipy as sc
import pyximport

pyximport.install()

from cython_module import hello_world
from Plot_funcs import plot_two_lines
from volterra import volterra_compute


def main():
    abscissa, numeric, analytic, p, force, P1, P2, absFp, argFp, U_analytic, U_numeric = volterra_compute()
    plot_two_lines(abscissa, numeric, analytic, 'Force', 't', 'P', 'Numeric', 'Analytic')
    plot_two_lines(abscissa, U_numeric, U_analytic, 'Displacement', 't', 'U', 'Numeric', 'Analytic')
    plot_two_lines(abscissa, numeric, P1, 'Force_old_1', 't', 'P1', 'Numeric', 'Analytic')
    plot_two_lines(abscissa, numeric, P2, 'Force_old_2', 't', 'P1', 'Numeric', 'Analytic')
    plot_two_lines(abscissa, p, force, 'Force_external', 't', 'Force', 'p', 'force')
    f_force = np.zeros_like(force, dtype='float64') + force
    #f_force[0] = f_force[1]
    #F_f = np.fft.fft(f_force)
    F_f = np.fft.fft(f_force)
    plot_two_lines(abscissa, absFp, argFp,
                   'Force_abs_angle', 't', 'Force', 'abs',
                   'angle')
    np.savetxt("F_force.csv", F_f, delimiter=';')
    np.savetxt("abs_F_force.csv", np.abs(F_f), delimiter=';')
    np.savetxt("arg_F_force.csv", np.angle(F_f), delimiter=';')
    #t, Fp, Fft_p = check_fourier_transform()
    #plot_two_lines(t, Fp, Fft_p, 'Forier_transform', 't', 'F(p)', 'F(p)', 'Fft(p)')
    #input()


def check_fourier_transform():
    N = 10000
    t = np.linspace(0, 100, N)
    p = np.zeros_like(t)
    Fp = np.zeros_like(t)
    p += 10
    p[0] = 1/(t[1]-t[2])
    Fp += 1/(2*np.pi)
    Fp[0] += 10*np.sqrt(2*np.pi)/(t[1]-t[0])
    Fft_p = np.abs(np.fft.fft(p))/(N)#*np.sqrt(2*np.pi))
    Fft_p[0] = Fft_p[1]
    Fp[0] = Fp[1]
    return t, Fp, Fft_p


def initial_fold(fold_name: str = 'Plots') -> None:
    if not os.path.exists(fold_name):
        os.mkdir(fold_name)
    else:
        pass
        #shutil.rmtree(fold_name)
        #os.mkdir(fold_name)


if __name__ == '__main__':
    initial_fold('Plots')
    main()
