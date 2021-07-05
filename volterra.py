import numpy as np
import scipy.linalg as la
import tqdm
from cython_module import calc_matrices_1, calc_p1, calc_P0, calc_v, calc_matrices_0, calc_P1, \
    calc_P2, calc_p0, harmonic_force, right_side_for_harmonic_force


def params():
    M1 = 5.
    K1 = 1.5
    a = 0.01
    v = -0.5
    T = 350
    h1 = 25000
    g = 5.
    M1, K1, a, v, T, h1, g = input_parameter(M1, K1, a, v, T, h1, g)
    return M1, K1, a, v, T, h1, g


def volterra_compute():
    M1, K1, a, v, T, h1, g = params()
    ampl_v = 1
    j = 1
    h = T / h1
    t = np.linspace(0, T, int(round(T / h)), dtype='float64')
    v_a = calc_v(t, v, ampl_v, a)
    if K1 < 0:
        print(f" v^2={v * v}; a={a}; K= {K1}; M={M1}\n  v^2<={1 - K1 * K1 / 4}")
    else:
        print(f" v^2={v * v}; a={a}; K= {K1}; M={M1}\n v^2<{1 - K1 / (M1)}")
    #if M1 != 0:
    #    Matr0 = calc_matrices_0(t, M1, v_a, h)
    #    p0 = calc_p1(t, 0, M1, g)
    #else:
    #    Matr0 = np.zeros_like(t, dtype='float64')
    #    p0 = np.zeros_like(t, dtype='float64')
    Matr = calc_matrices_1(t, K1, M1, v_a, h)
    #p = calc_p1(t, K1, M1, g)
    p = right_side_for_harmonic_force(g, 0, t, K1, M1)
    force = harmonic_force(g, t, 0, True)
    #P0 = calc_P0(t, M1, v_a, g, j)
    P1 = calc_P1(t, K1, M1, v_a, force, j)
    #P2 = calc_P2(t, K1, M1, v_a, g, j)
    sol = la.solve_triangular(Matr, p, 0, True, False, False, None, False)
    #if M1 != 0:
    #    sol0 = la.solve_triangular(Matr0, p0, 0, True, False, False, None, False)
    #else:
    #    sol0 = np.zeros_like(t, dtype='float64')
    return t, sol, P1




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