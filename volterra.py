import numpy as np
import scipy.linalg as la
import tqdm
import scipy.integrate as integrate
from cmath import sqrt, cos, sin, cosh, sinh, pi
from cython_module import calc_matrices_1, calc_p1, calc_P0, calc_v, calc_matrices_0, calc_P1, \
    calc_P, calc_p0, harmonic_force, right_side_for_harmonic_force, integrand, calc_P2, SystemUnderStudy,\
    calc_U_analityc, calc_U
import time


def params():
    M1 = 10.
    K1 = -0.5
    a = 0.01
    v = 0.1
    T = 100
    h1 = 20000
    g = 10.
    freq = 0#3
    amplitude = 0
    phase = 0#pi/4.5
    # M1, K1, a, v, T, h1, g = input_parameter(M1, K1, a, v, T, h1, g)
    return M1, K1, a, v, T, h1, g, freq, amplitude, phase


def volterra_compute():
    M1, K1, a, v, T, h1, g, freq, amplitude, phase = params()
    syst = SystemUnderStudy(M1, K1, a, v, T, h1, g, freq, amplitude, phase)
    ampl_v = 1
    j = 1
    h = T / h1
    t = np.linspace(0, T, int(round(T / h)), dtype='float64')
    v_a = calc_v(t, v, ampl_v, a)
    if K1 < 0:
        print(f" v^2={v * v}; a={a}; K= {K1}; M={M1}\n  v^2<={1 - K1 * K1 / 4}")
    else:
        print(f" v^2={v * v}; a={a}; K= {K1}; M={M1}\n v^2<{1 - K1 / (M1)}")
    time1 = time.time()
    Matr = calc_matrices_1(t, K1, M1, v_a, h)
    print(f'time work calc_matrices_1 is {time.time() - time1} sec')
    force = harmonic_force(g, t, phase, True, freq, amplitude)
    if M1 != 0:
        p = right_side_for_harmonic_force(g, phase, freq, amplitude, t, K1, M1, h1)
    else:
        p = force
    print(p)

    P, absFp, argFp = calc_P(syst, t, K1, M1, v_a, force, j, freq, amplitude, phase, g)
    P1 = calc_P1(t, K1, M1, v_a, g, j)
    P2 = calc_P2(t, K1, M1, v_a, g, j)
    print(P1)
    sol = la.solve_triangular(Matr, p, 0, True, False, False, None, False)
    U_analytic = calc_U_analityc(t, K1, M1, v_a, force, j, freq, amplitude, phase, g)
    U_numeric = calc_U(t, K1, M1, sol, force, v_a)
    return t, sol, P, p, force, P1, P2, absFp, argFp, U_analytic, U_numeric


def right_side_for_harmonic_force_1(g, phase, time, K, M):
    max_iter = 10000
    freq = np.sqrt(K / M)
    rs = np.zeros_like(time)
    #for i in tqdm.tqdm(range(1, time.shape[0])):
    for i in range(1, time.shape[0]):
        time_max = time[i]
        #tau = np.linspace(0,time_max, max_iter)
        #intag = np.zeros(max_iter, dtype='float64')
        print(f'g= {g}, time_max={time_max}, freq={freq}, phase={phase}')
        print(f'Types g= {type(g)}, time_max={type(time_max)}, freq={type(freq)}, phase={type(phase)}')
        res = integrate.quad(lambda x: integrand(g, time_max, freq, phase, 0,0, x), 0., \
              float(time_max))
        print(res)
        rs[i] = res[0]
        #rs[i]=_right_side_for_harmonic_force_(g, phase, time_max, tau,max_iter, freq, intag)*2/M+\
        #      2 * cos(freq * time_max) / M
        rs[i]+= cos(freq*time_max)
        rs[i] *= 2/M
    return rs



def input_parameter(M1, K1, a, v, T, h1, g):
    M2 = input(f"Введите значение M ({M1} по умолчанию)\n M=")
    if M2 != '':
        M1 = float(M2)
    print(f"-2<K<{M1}")
    if M1 == 0:
        K1 = -0.5
    K2 = input(f"Введите значение K ({K1} по умолчанию)\n K=")
    if K2 != '':
        if M1 > 0:
            K1 = float(K2)
        else:
            if float(K2) >= 0:
                K2 = input(
                    f"ERROR!!!\n При M=0, K<0!\n Введите значение K ({K1} по умолчанию)\n K=")
                K1 = float(K2)
            else:
                K1 = float(K2)
    if K1 < 0:
        print(f"\n v^2<={1 - K1 * K1 / 4} \n")
    else:
        print(f"\n v^2<{1 - K1 / M1} \n")
    v2 = input(f"Введите значение v ({v} по умолчанию)\n v=")
    if v2 != '':
        v = float(v2)

    a2 = input(f"Введите значение a ({a} по умолчанию)\n a=")
    if a2 != '':
        a = float(a2)
    if a == 0:
        T = 100
    T2 = input(f"Введите значение T ({T} по умолчанию)\n T=")
    if T2 != '':
        if float(T2) > 0:
            T = float(T2)

    h2 = input(f"Введите количество элементов разбиения ({h1} по умолчанию) \n h=")
    if h2 != '':
        h1 = float(h2)
    print("Считаю")
    return M1, K1, a, v, T, h1, g
