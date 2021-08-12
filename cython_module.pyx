from libc.math cimport sqrt, pow, exp,cos,sin, cosh, sinh, fabs,pi, atan, log, atan
cimport cython
from cython.parallel cimport prange
from libc.stdio cimport printf
import tqdm
import numpy as np
#import scipy.special as sc
#import scipy
cimport scipy.special.cython_special as csc
from c_module cimport Bessel_c, summand_c, integral_c

#!-*-coding: utf-8 -*-
cdef class SystemUnderStudy(object):
    cdef public:
        double M, K, a, v, T, h, g, freq, ampl, phase
    def __init__(self, M1, K1, a1, v11, T1, h1, g1, freq1, amplitude1, phase1):
        self.M = M1
        self.K = K1
        self.a = a1
        self.v = v11
        self.T = T1
        self.h = h1
        self.g = g1
        self.freq = freq1
        self.ampl = amplitude1
        self.phase = phase1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double ext_force(self, double tau)nogil:
        cdef double term_1, term_2, term_3, delta
        term_1 = 0
        if tau == 0:
            term_1 = self.h/self.T
        term_2 = 0
        term_3 = 0
        if tau >=0 :
            term_2 = self.g
            term_3 = self.ampl * sin(self.freq*tau + self.phase)
        return term_1+term_2+term_3

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double d_ext_force(self, double tau)nogil:
        cdef double term_1, term_2, delta
        term_1 = 0
        term_2 = 0
        if tau == 0:
            delta = self.h/self.T
            term_1 = (self.g + self.ampl*sin(self.phase))*delta
        if tau>=0:
            term_2 = self.ampl*self.freq*cos(self.freq*tau+self.phase)
        return term_1+term_2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double d2_ext_force(self, double tau)nogil:
        cdef double term_1, term_2, delta
        term_1 = 0
        term_2 = 0
        if tau == 0:
            delta = self.h / self.T
            term_1 =  self.ampl * self.freq * cos(self.phase) * delta
        if tau >= 0:
            term_2 = -self.ampl * (self.freq**2) * sin(self.freq * tau + self.phase)
        return term_1 + term_2
"""
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double velocity(self, tau)nogil:
        cdef double vel
        vel = self.v + self.a*tau
        return vel

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double d_velocity(self, tau)nogil:
        cdef double vel
        vel = self.a
        return vel

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double d2_velocity(self, tau)nogil:
        cdef double vel
        vel = 0.
        return vel
"""



def hello_world():
    _hello_world_()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _hello_world_()nogil:
    printf("hello_world")


def harmonic_force(g: float, time: np.ndarray, phase: float = 0., delta_indicator: bool = False,
                   freq: float = 0, amplitude: float = 0) -> np.ndarray:
    delta = float(int(delta_indicator)/(time[1]-time[0]))
    force = np.zeros_like(time, dtype = 'float64')
    max_iter = int(force.shape[0])
    _harmonic_force_(g, phase, delta, time, force, freq, amplitude, max_iter)
    return force


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _harmonic_force_(double g, double phase,
                           double delta, double[:] time, double[:] force, double freq,
                           double amplitude, int iter_max)nogil:
    cdef int i
    cdef double term_1, term_2, term_3
    for i in prange(iter_max, nogil=True):

        term_1 = 0
        if time[i] == 0:
            term_1 = delta
        term_2 = 0
        term_3 = 0
        if time[i] >= 0:
            term_2 = g
            term_3 = amplitude * sin(freq * time[i] + phase)

        force[i] =term_1 + term_2 + term_3   #g + amplitude * sin( freq* time[i] + phase)
    #force[0] += delta


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double amplitude_force(double time, double amplitude)nogil:
    return amplitude*1



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double frequency_force(double time, double freq)nogil:
    return freq*1 #* cos(time / 100)


def right_side_for_harmonic_force(g: float, phase: float, freq_f: float, amplitude: float,
                                  time: np.ndarray, K: float, M: float, h1: int) -> np.ndarray:
    max_iter = 10#*h1
    assert M>0, 'In func "right_side_for_harmonic_force" parametr M can\'t be zero'
    rs = np.zeros_like(time, dtype='float64')
    if K>= 0:
        positive_K(rs, time, max_iter, amplitude, freq_f, K, M, phase,g)
    else:
        negative_K(rs, time, max_iter, amplitude, freq_f, K, M, phase,g)
    return rs


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void negative_K(double[:] rs, double[:] time, int max_iter,double amplitude, double bar_omega, double K, double M,
                                  double phase, double g):
    freq = sqrt(abs(K / M))
    for i in range(1, time.shape[0]):
        time_max = time[i]
        integ = 0
        d_tau = time_max / max_iter
        rs[i] = integrated_expression_negative(amplitude, bar_omega, K, M, time[i], phase)
        rs[i] += g * sinh(freq * time_max) / freq
        rs[i] += cosh(freq * time_max)
        rs[i] *= 2 / M


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double integrated_expression_negative(double A, double bar_omega, double K, double M, double t,
                                           double phi)nogil:
    cdef:
        double term_1, term_2, KM, denominator, term_3
    KM = sqrt(-K/M)
    denominator = bar_omega**2+KM**2

    term_1 = bar_omega*cos(phi)*cosh(KM*t)
    term_2 = -bar_omega*cos(bar_omega*t+phi)
    term_3 = KM*sin(phi)*sinh(KM*t)

    return A*(term_1+term_2+term_3)/denominator


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void positive_K(double[:] rs, double[:] time, int max_iter,double amplitude, double bar_omega, double K, double M,
                                  double phase, double g):
    cdef int i
    freq = sqrt(abs(K / M))
    for i in range(1, time.shape[0]):
        time_max = time[i]
        integ = 0
        d_tau = time_max/max_iter
        rs[i] = integrated_expression_positive(amplitude, bar_omega, K, M, time[i], phase)
        rs[i] += g * sin(freq * time_max) / freq
        rs[i] += cos(freq*time_max)
        rs[i] *= 2/M



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double integrated_expression_positive(double A, double bar_omega, double K, double M, double t,
                                           double phi)nogil:
    cdef:
        double term_1, term_2, KM, denominator, term_3
    KM = sqrt(K / M)
    denominator = bar_omega ** 2 - KM ** 2

    term_1 = bar_omega * cos(phi) * cos(KM * t)
    term_2 = -bar_omega * cos(bar_omega * t + phi)
    term_3 = -KM * sin(phi) * sin(KM * t)

    return A * (term_1 + term_2 + term_3) / denominator


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _right_side_for_harmonic_force_(double g, double phase, double freq_f,
                                            double amplitude, double time, double d_tau,
                                            int max_iter, double freq)nogil:
    cdef:
        int i, j
        double step_1, step_2,  step_time, rs
    step_1 = 0.
    step_2 = 0.
    step_time = d_tau#tau[1]-tau[0]
    rs = 0.
    #tau_1 = 0
    #tau_2 = d_tau
    for i in prange(1, max_iter, nogil=True):
        step_2 = _integrand_(g, time, freq, phase, freq_f, amplitude, d_tau*(i-1))#[i - 1])
        step_1 = _integrand_(g, time, freq, phase, freq_f, amplitude, d_tau*i)#[i])
        step_time = d_tau#[i] - tau[i-1]
        rs += (step_1+step_2)*step_time/2
        step_2 = 0.#step_1
        step_1 = 0.
    step_1 = 0.
    step_2 = 0.
    #for i in prange(1, max_iter, nogil=True):
    #    ret += rs[i]
    return rs#ret#


def integrand(g, time, freq, phase, freq_f, amplitude, tau):
    _integrand_(g, time, freq, phase, freq_f, amplitude, tau)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _integrand_(double g, double time_max, double freq_cos, double phase, double freq_f,
                        double amplitude, double tau)nogil:
    cdef double step
    step = 0.
    #step = g + amplitude_force(tau) * sin(frequency_force(tau)* tau + phase)
    step = amplitude_force(tau, amplitude) * sin(frequency_force(tau, freq_f)* tau + phase)
    step = step * cos(freq_cos*(time_max-tau))
    return step




def calc_v(t, v, ampl_v, a):
    out_v=np.zeros_like(t, dtype='float64')
    iter_max = t.shape[0]
    _calc_v(t,out_v, v, ampl_v, a, iter_max)
    return out_v


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calc_v(double[:] t, double[:] out_v, double v, double ampl_v, double a,
                  int iter_max) nogil:
    cdef int i
    cdef double h
    h=t[1]-t[0]
    for i in prange(iter_max, nogil=True):
        out_v[i]=v+ampl_v*a*t[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double integral(double [:] O, int i, double h) nogil:
    cdef int j
    cdef double su
    su=0.
    j=1
    while j<=i:
        su=su+0.5*(O[j]+O[j-1])*h
        j=j+1
    return su

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double integral_O_M_eq_0(double v0,double t,double v_i,double v_j, int i, double h,
                              double K) nogil:
    cdef double su, a,b,s,v
    a=(v_i-v_j)/h
    b=1-(K**2)/4
    v=v0+a*t
    if a!=0:
        s=sqrt(b-v**2)
        su=v*s-v0*sqrt(b-v0**2)+b*(atan(v/s)-atan(v0/sqrt(b-v0**2)))
        su=su/(2*a)
    else:
        su=sqrt(b-v**2)*t
        #printf('0\n')
    return su

#It function calculate Bessel function and its multiplier
@cython.boundscheck(False)
@cython.wraparound(False)
cdef long double Bessel(double K, double M, long double ti, long double tj, long double v,
                        long double x_i,long double x_j) nogil:
    cdef long double ko, Be
    ko=(ti-tj)**2-(x_i-x_j)**2
    if (ko>0.)&(M>0.):
        Be=(ti-tj-(x_i-x_j)*v)*csc.j1(sqrt(ko))/sqrt(ko)
    elif (ko==0)&(M>0):
        Be=(ti-tj-(x_i-x_j)*v)
    elif(ko==0.)&(M==0.)&(K<0):
        Be=fabs(K)/2.
    elif (ko>=0.)&(M==0.)&(K<0):
        Be=(-K)*csc.j0(sqrt(ko))/2
    elif (ko<0.)&(M>=0.):
        Be=0.
    else:
        printf('Error! Failed parametrs! Func Bessel')
        return 0.
    return Be

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long double summand(long double ti,long double tj, long double K, long double M) nogil:
    cdef long double S
    if (K>0.)&(M>0):
        S=2.*cos((ti-tj)*sqrt(K/M))/M
    elif (K<0.)&(M>0):
        S=2.*cosh((ti-tj)*sqrt(-K/M))/M
    elif (K==0.)&(M>0):
        S=2./M
    elif (K<0)&(M==0):
        S=0.
    return S

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long double integral_0(long double [:] v, int i, long double [:] t) nogil:
    cdef double s
    s= v[0]*t[i]
    s+= t[i]*t[i]*(  (v[i]-v[i-1])/(t[i]-t[i-1])  )/2
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void O(double [:] Omega, double [:] v, double M, double K) nogil:
    cdef int i, len_vec
    len_vec = len(v)
    i=0
    if M!=0:
        #for i in prange(v.shape[0], nogil=True):
        for i in prange(len_vec, nogil=True):
            Omega[i]=sqrt((K*M+2*(sqrt((1-v[i]*v[i])*M*M-K*M+1)-1)))/M
    else:
        printf('M=0')
        for i in prange(v.shape[0], nogil=True):
            Omega[i]=sqrt(1-v[i]*v[i]-K*K/4)


def calc_matrices_0(t, M, v,h):
    out = np.zeros((t.shape[0],t.shape[0]), dtype='float64')
    _parallel_calc_matrices_0(t, out, M, v,h)
    return out


#==========Calculation extreme case=============
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _parallel_calc_matrices_0(long double[:] t, long double[:,:] out,
                                  long double M, long double[:] v, long double h) nogil:
    cdef  int i, j, l
    cdef long double x_i, x_j,J
    j=0
    i=0
    l=len(t)//50
    printf('\n Calculate matrices: \n')
    printf('___________________________________________________\n')
    for i in prange(t.shape[0],nogil=True):
        if (i % l) == 0:
            printf('#')
        x_i=integral_0(v,i,t)
        for j in range(i+1):
            x_j=integral_0(v,j,t)
            J=Bessel(0,M,t[i],t[j],v[i],x_i,x_j)
            out[i,j]=-(J-summand(t[i],t[j],0.,M))*h
    printf('#\n')
    out[0,0]/=2
    out[0,0]+=1
    for i in range(1,t.shape[0]):
        out[i,i]=out[i,i]/2.+1.
        out[0,i]/=2.


def calc_matrices_1(t, K, M, v, h):
    out = np.zeros((t.shape[0],t.shape[0]), dtype='float64')
    t_len = t.shape[0]
    _parallel_calc_matrices_1(t, out, K, M, v, h, t_len)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _parallel_calc_matrices_1(long double[:] t, long double[:,:] out, long double K,
                                  long double M, long double[:] v, long double h, long int t_len) nogil:
    cdef  int i, j, l
    cdef long double x_i, x_j,J, v00
    v00 = v[0]
    j=0
    i=0
    l=t_len//50
    printf("\n Calculate matrices: \n")
    printf("___________________________________________________\n")
    for i in prange(t_len, nogil=True):
        if (i%l)==0:
            printf("#")
        #x_i = integral_0(v,i,t)
        x_i = integral_c(v[i],v[i-1], v00, t[i], t[i-1])
        for j in range(i+1):
            #x_j = integral_0(v,j,t)
            x_j = integral_c(v[j],v[j-1], v00, t[j], t[j-1])
            J = Bessel_c(K,M,t[i],t[j],v[i],x_i,x_j)
            out[i,j]=-(J-summand_c(t[i],t[j],K,M))*h
    printf("#\n")
    out[0,0]/=2
    out[0,0]+=1
    for i in range(1,t.shape[0]):
        out[i,i]=out[i,i]/2.+1.
        out[0,i]/=2.


def calc_p0(t,M):
    out = np.zeros_like(t, dtype='float64')
    _calc_p0(t, out, M)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calc_p0(double[:] t, double[:] out, double M)nogil:
    cdef int i,len_vec
    len_vec = len(t)
    i=0
    #for i in range(t.shape[0]):
    for i in prange(len_vec, nogil=True):
        out[i]=(2./M)*10*t[i]


def calc_p1(t,K, M,g):
    out = np.zeros_like(t, dtype='float64')
    _calc_p1(t, out, K, M,g)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calc_p1(long double[:] t, long double[:] out, long double K, long double M,
                   long double g)nogil:
    cdef int i, flag
    cdef long double K1, M1
    flag=0
    if (K>0.)&(M>0.):
        i=0; K1=K;M1=M;flag=1
        for i in range(t.shape[0]):
            out[i]=2.*g*sin(t[i]*sqrt(K1/M1))/sqrt(K1*M1)
    elif (K<0.)&(M>0.):
        i=0; K1=K;M1=M;flag=2
        for i in range(t.shape[0]):
            out[i]=2.*g*sinh(t[i]*sqrt(-K1/M1))/sqrt(-K1*M1)
    elif (K==0.)&(M>0.):
        i=0; K1=K;M1=M;flag=3
        for i in range(t.shape[0]):
            out[i]=(2./M1)*g*t[i]
    elif (K<0.)&(M==0.):
        flag=4
        for i in range(t.shape[0]):
            out[i]=g
        out[0]=g/2
    else:
        printf('\n Error! Failed parametrs! Func _calc_p1: \n K= %f, M=%f ; cicle number (%d) \n', K, M,flag)


def calc_P0(t, M, v,g,j):
    outP0 = np.zeros_like(t, dtype='float64')
    #omega =np.zeros_like(t, dtype='float64')
    U =np.zeros_like(t, dtype='float64')
    O =np.zeros_like(t, dtype='float64')
    _calc_P0(t, outP0, M, v,g,j,U,O)
    return outP0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calc_P0(double[:] t, double [:] outP0, double M,  double[:] v, double g, int j,  double[:] U,double [:] omega) nogil:
    cdef int i,l
    cdef double absFp, argFp,C
    l=len(t)//50
    O(omega,v,M,0)
    absFp=g#/omega[0]
    argFp=pi/2
    C=M*sqrt(omega[0])/sqrt((1-v[0]*v[0])*(M*M*omega[0]*omega[0]+2))*omega[0]
    printf('\n Calculate Analityc \n');    printf('___________________________________________________\n')
    i=0
    for i in prange(t.shape[0]):
        if (i%l)==0:
            printf('*')
        U[i]=C*M*sqrt((1-v[i]**2)*omega[i]**3)/sqrt(M**2*omega[i]**3+2)
        outP0[i]=g+U[i]*absFp*sin(integral(omega, i, t[i]-t[i-1])-argFp)
    printf('*\n')


def calc_P1(t, K1, M, v,force,j):
    outP0 = np.zeros_like(t, dtype='float64')
    #omega =np.zeros_like(t, dtype='float64')
    U =np.zeros_like(t, dtype='float64')
    O =np.zeros_like(t, dtype='float64')
    #four_force= np.fft.fft(force)
    #absFp = np.abs(four_force)
    #argFp = np.angle(four_force)#-pi/2
    _calc_P1(t, outP0, K1, M, v,force,j,U,O)

    return outP0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calc_P1(double[:] t, double [:] outP0, double K, double M, double[:] v, double p,
                                                                                            int j,
                   double[:] U,double [:] omega) nogil:
    cdef int i,l
    cdef double C,integ,O0, absFp, argFp
    l=len(t)//50
    O(omega,v,M,K)
    printf('O(0)=%f',omega[0])
    absFp=p/(omega[0])
    argFp=pi/2
    C=sqrt(sqrt(1.-v[0]*v[0]-omega[0]*omega[0])/(omega[0]*(1.+M*sqrt(1.-v[0]**2-omega[0]*omega[0]) )))
    printf('\n Calculate Analityc, C= %f\n',C);    printf('___________________________________________________\n')
    i=0
    for i in prange(t.shape[0], nogil=True):
        if (i%l)==0:
            printf('*')
        if M!=0:
            integ=integral(omega, i, t[i]-t[i-1])
        else:
            integ=integral_O_M_eq_0(v[0],t[i],v[i],v[i-1],i,t[i+1]-t[i],K)
        O0=K*p/(2.*sqrt(1.-v[i]**2)+K)
        U[i]=C*sqrt(sqrt(1.-v[i]*v[i]-omega[i]*omega[i])/
                    (omega[i]*(1.+M*sqrt(1.-v[i]*v[i]-omega[i]*omega[i]))))*(M*omega[i]*omega[i]-K)
        outP0[i]=p-O0+U[i]*absFp*sin(integ-argFp)

    printf('*\n')


def calc_P(syst: SystemUnderStudy, t: np.ndarray, K: float, M: float, v: np.ndarray, force: np.ndarray, j: int, freq: float,
           amplitude: float, phase: float, g: float) -> np.ndarray:
    return_force = np.zeros_like(t, dtype='float64')
    #omega =np.zeros_like(t, dtype='float64')
    U =np.zeros_like(t, dtype='float64')
    O =np.zeros_like(t, dtype='float64')
    O_new(O, v*v, M, K)
    print(f'O(0)={O[0]}')
    ff = fourier_force(g, O[0], ampl=amplitude, freq=freq, phase=phase, step_time=t[1]-t[0])
    absFp = np.zeros_like(force) + np.abs(ff*np.sqrt(2*pi) ) #  g /O[0]  #
    argFp = np.zeros_like(force) + np.angle(ff*np.sqrt(2*pi) )  #  pi/2  #
    #for i in range(t.shape[0]):
    #    ff = fourier_force(g, O[i], amplitude, freq, phase, t[1]-t[0])
    #    ff_abs, ff_arg = abs_arg(ff)
    #    absFp[i] = np.abs(ff*np.sqrt(2*pi) ) #10  # np.abs(four_force)
    #    argFp[i] = np.angle(ff*np.sqrt(2*pi) )  #pi/2  # np.angle(four_force)# + pi/4
    #absFp = np.zeros_like(force) + np.abs(ff*np.sqrt(2*pi) ) #10  # np.abs(four_force)
    #argFp = np.zeros_like(force) - np.angle(ff*np.sqrt(2*pi) )  #pi/2  # np.angle(four_force)# + pi/4
    #absFp = np.abs(four_force)
    #argFp = np.angle(four_force) - pi/2  #+ pi/4#
    _calc_P(syst, t, return_force, K, M, v*v,force,j, absFp, argFp, U,O, amplitude, freq, phase)
    return return_force, absFp, argFp


@cython.boundscheck(False)
@cython.wraparound(False)
cdef (double, double) abs_arg(double complex f)nogil:
    cdef double abs_f, arg_f
    abs_f = 0.
    arg_f = 0.
    abs_f = sqrt(f.real**2+f.imag**2)*sqrt(2*pi)
    arg_f = atan(f.imag/f.real)
    return abs_f, arg_f


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calc_P(SystemUnderStudy syst, double[:] t, double [:] outP0, double K, double M, double[:] v_sq, double[:] p,
                                                                                            int j,
                   double[:] absFp, double[:] argFp, double[:] U,double [:] omega,
                  double amplitude, double freq, double phase) nogil:
    cdef:
        int i,l
        double C,integ_omega, C1, C2, omega_0, v_sq_0, C_old, g
        double term_2, term_3, multiplier_1, multiplier_2, absFp_old, argFp_old
        double complex F_p
    l=len(t)//50
    g = syst.g
    #O_new(omega,v_sq,M,K)

    absFp_old=g/(omega[0])
    argFp_old=pi/2
    C_old = sqrt(
            sqrt(1.-v_sq[0]-omega[0]*omega[0])/
           (omega[0]*( 1.+M*sqrt(1.-v_sq[0]-omega[0]*omega[0]) ))
    )
    # По идее, в этой константе и должен быть модуль фурье-образа силы
    omega_0 = omega[0]
    v_sq_0 = v_sq[0]
    C1 = sqrt(
              omega_0 * (1 - v_sq_0 - omega_0**2) * (M*M*omega_0**2 - K*M + 2)
             )
    C2 = omega_0 * sqrt( M*omega_0**2 - K )*(1 + M * sqrt( 1 - v_sq_0 - omega_0**2))
    C = C1 / C2
    printf('\n Calculate Analityc, C= %f, C_old= %f, C2= %f, v^2= %f\n',C, C_old,
           M * sqrt( 1 - v_sq_0 - omega_0**2), v_sq_0)
    printf('___________________________________________________\n')
    for i in prange(t.shape[0], nogil=True):
        if (i%l)==0:
            printf('*')
        if M!=0:
            integ_omega = integral(omega, i, t[i]-t[i-1])
        else:
            integ_omega = integral_O_M_eq_0(sqrt(v_sq[0]),t[i], sqrt(v_sq[i]),sqrt(v_sq[i-1]), i, t[i+1]-t[i],K)
        #F_p = fourier_force(g, omega[i], amplitude, freq, phase, t[1] - t[0])
        #absFp_old, argFp_old = abs_arg(F_p)
        outP0[i] = step_cicle_for_force(syst, omega[i], K, M, p[i], v_sq[i], absFp[i], argFp[i], C,
                                        integ_omega, t[i])
    printf('*\n')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double step_cicle_for_force(SystemUnderStudy syst, double omega, double K, double M, double p, double v_sq,
                                 double absFp, double argFp, double C, double integ_omega, double time)nogil:
    cdef:
        int i
        double term_2, term_3, multiplier_1, multiplier_2, outP0,z_fr_f, p1
    p1 = syst.ext_force(time)
    term_2 = K*p1 / (2. * sqrt(1. - v_sq) + K)
    z_fr_f = zero_freq_force(syst, time, v_sq)
    multiplier_1 =  sqrt(
            (M*omega*omega-K)/
            (omega*(M*M*omega*omega-K*M+2))
        )*(M*omega*omega-K)
    multiplier_2 = sin(integ_omega - argFp)
    term_3 = C * multiplier_1 * multiplier_2 * absFp
    outP0 = p1 - term_2 + term_3 - M*z_fr_f
    return outP0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double zero_freq_force(SystemUnderStudy syst, double time, double v_sq)nogil:
    cdef double term_1, zn_1
    zn_2 = 2*sqrt(1-v_sq)+syst.K
    term_1 = syst.d2_ext_force(time)/zn_2
    return term_1





@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex fourier_force(double g, double omega, double ampl, double freq, double phase,
                                  double step_time)nogil:
    cdef:
        double complex p, term_1, term_2, term_3, _sum_
    term_1 = 1./sqrt(2*pi)+0j
    term_2 =0. + g*1j/(omega*sqrt(2*pi))
    if omega == 0.:
        printf('Omega equal 0')
        term_2 += (1/step_time)*g*sqrt(pi/2) + 0j
    term_3 = (ampl/sqrt(2*pi))*(freq*cos(phase) - 1j*omega*sin(phase))/(freq**2-omega**2)
    if freq == omega:
        printf('Omega equal frequency force')
        term_3 += 0.5*sqrt(pi/2)*ampl*(sin(phase)+1j*cos(phase))/step_time
    if freq == -omega:
        printf('Omega equal -1*frequency force')
        term_3 += 0.5 * sqrt(pi / 2) * ampl * (sin(phase) - 1j * cos(phase)) / step_time
    #printf('term_1=%f+i%f, term_2=%f+i%f, term_3=%f+i%f\n',term_1.real,term_1.imag,term_2.real,
    #       term_2.imag, term_3.real, term_3.imag)
    _sum_ = term_1 + term_2 + term_3
    return _sum_


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cicle_for_force(int max_iter, int l, double[:] omega, double[:] t, double K, double M,
                          double[:] p, double[:] v_sq, double[:] absFp, double[:] argFp, double C,
                          double[:] outP0)nogil:
    cdef:
        int i
        double term_2, term_3, multiplier_1, multiplier_2, integ_omega
    for i in prange(max_iter, nogil=True):
        if (i%l)==0:
            printf('*')
        # Это не работает для M=0!!!
        integ_omega = integral(omega, i, t[i]-t[i-1])
        term_2 = K*p[i] / (2. * sqrt(1. - v_sq[i]) + K)
        multiplier_1 =  sqrt(
                                (M*omega[i]*omega[i]-K)/
                                (omega[i]*(M*M*omega[i]*omega[i]-K*M+2))
                            )*(M*omega[i]*omega[i]-K)
        #multiplier_1 = sqrt(
        #                   (M*omega[i]**2 - K) / \
        #                   omega[i]*(M*M * omega[i]*omega[i] - K*M + 2.)
        #               )
        #multiplier_1 = sqrt(
        #                    sqrt(1-v_sq[i]-omega[i]*omega[i])/\
        #                    (omega[i]*(M*M*omega[i]*omega[i]-K*M+2))
        #                    )
        multiplier_2 = sin(integ_omega - argFp[i])#*(M*omega[i]*omega[i]-K)
        #multiplier_2 = sin(integ_omega - argFp_old)
        term_3 = C * multiplier_1 * multiplier_2 * absFp[i]#*omega[i]
        #term_3 = C * multiplier_1 * multiplier_2 * absFp_old
        outP0[i] = p[i] - term_2 - term_3



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double moving_stationary_setting(double M, double K, double v_sq, double omega_0,
                                    double absFp, double argFp, double t)nogil:
    cdef double sqrt_div_freq, ret
    sqrt_div_freq = sqrt(1 - v_sq - omega_0)
    ret = sqrt_div_freq * absFp/(omega_0*( 1 + M*sqrt_div_freq ))
    ret *= sin(omega_0*t-argFp)
    return ret



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void O_new(double [:] Omega, double [:] v_sq, double M, double K) nogil:
    cdef:
        int i, len_vec
        double KM, KK, MM
    len_vec = len(v_sq)
    i=0
    if M!=0:
        KM = K*M
        MM = M*M
        for i in prange(len_vec, nogil=True):
            Omega[i]=sqrt((KM+2*(sqrt((1-v_sq[i])*MM-KM+1)-1)))/M
    else:
        printf('M=0')
        KK=K*K
        for i in prange(len_vec, nogil=True):
            Omega[i]=sqrt(1-v_sq[i]-KK/4)


def calc_P2(t, K1, M, v,g,j):
    outP0 = np.zeros_like(t, dtype='float64')
    #omega =np.zeros_like(t, dtype='float64')
    U =np.zeros_like(t, dtype='float64')
    O =np.zeros_like(t, dtype='float64')
    _calc_P2(t, outP0, K1, M, v,g,j,U,O)
    return outP0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calc_P2(double[:] t, double [:] outP0, double K1, double M,  double[:] v, double g, int j,  double[:] U,double [:] omega) nogil:
    cdef int i,l
    cdef double absFp, argFp,C, integ, O0
    l=len(t)//50
    O(omega,v,M,K1)
    absFp=g#/omega[0]
    argFp=pi/2
    if M!=0:
        C=(sqrt(1-v[0]*v[0]-omega[0]*omega[0])/(omega[0]*(1+M*sqrt(1-v[0]*v[0]-omega[0]*omega[0]))))/sqrt((M*omega[0]*omega[0]-K1)/(omega[0]*(M*M*omega[0]*omega[0]-K1*M+2)))/omega[0]
    else:
        C=sqrt(    2*(1-v[0]**2-omega[0]**2)/(fabs(K1)*omega[0])    )
    printf('\n Calculate Analityc, for P2 C= %f\n',C)
    printf('___________________________________________________\n')
    i=0
    for i in prange(t.shape[0]):
        if (i%l)==0:
            printf('*')
        if M!=0:
            integ=integral(omega, i, t[i]-t[i-1])
        else:
            integ=integral_O_M_eq_0(v[0],t[i],v[i],v[i-1],i,t[i]-t[i-1],K1)
        U[i]=C*sqrt((M*omega[i]*omega[i]-K1)/(omega[i]*(M*M*omega[i]*omega[i]-K1*M+2)))*(M*omega[i]*omega[i]-K1)
        O0=K1*g/(2*sqrt(1-v[i]**2)+K1)
        outP0[i]=g-O0+U[i]*absFp*sin(integ-argFp)
    printf('*\n')











'''    
def calc_p2(t,K1,M,g):
    out = np.zeros_like(t, dtype='float64')
    _calc_p2(t, out, K1, M,g)
    return out
'''




