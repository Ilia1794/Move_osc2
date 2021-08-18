from libc.math cimport sqrt, pow, exp,cos,sin, cosh, sinh, fabs,pi, atan, log, atan
cimport cython
from cython.parallel cimport prange
from libc.stdio cimport printf
import tqdm
import numpy as np
#import scipy.special as sc
#import scipy
cimport scipy.special.cython_special as csc

#from c_module cimport Bessel_c, summand_c, integral_c

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
    cdef double ext_force(self, double tau):
        cdef double term_1, term_2, delta, term_3_r
        cdef double complex term_3
        term_1 = 0
        term_2 = 0
        term_3_r = 0
        if tau == 0:
            term_1 += self.h/self.T
            term_2 += self.g / 2
            term_3 = self.ampl * np.exp(-1j*self.freq * tau) / 2
            term_3_r +=term_3.real
        if tau >0 :
            term_2 = self.g
            term_3 = self.ampl * np.exp(-1j * self.freq * tau)
            term_3_r += term_3.real
        return term_1+term_2+term_3_r

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double sum_force_f(self, double tau, double v_sq):
        cdef:
            double zero_freq, harmonic_force, zn_1, zn_2, K, M,omega, g, A
            double complex h_f_ch, h_f_zn, h_f
        K, M,omega, g, A = self.K, self.M, self.freq,self.g, self.ampl
        zn_1 = 2*sqrt(1-v_sq)+self.K
        zero_freq = 0
        if tau == 0:
            zero_freq = self.h/self.T
        zero_freq +=g*K / zn_1
        if fabs(omega)<=sqrt(1-v_sq):
            #zn_2 = 2*sqrt(1-v_sq-(omega**2))+ K - M*(omega**2)
            #harmonic_force = A * cos(omega * tau) * (K - M * omega ** 2) / zn_2
            h_f_ch = A * np.exp(-1j * omega * tau)
            h_f_zn = 2*sqrt(1-v_sq-(omega**2))+ K - M*(omega**2)
            h_f = h_f_ch / h_f_zn
            harmonic_force = h_f.real
        else:
            h_f_ch = A*np.exp(-1j*omega*tau)
            h_f_zn = (2j*self.sign(omega)*sqrt(v_sq+omega**2-1)
                      - K + M*omega**2)
            h_f = h_f_ch/h_f_zn
            harmonic_force = h_f.real
            #zn_2 = (2*self.sign(omega)*sqrt(fabs(1-v_sq-omega**2))
            #         - K + M*omega**2)
            #harmonic_force = A*sin(omega*tau)*(K- M*omega**2 )/zn_2
        return zero_freq + harmonic_force


    cdef double sign(self, double omega)nogil:
        return omega/fabs(omega)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double sum_force_d(self, double tau, double v_sq)nogil:
        cdef:
            double zero_freq, harmonic_force, zn_1, zn_2
        zn_1 = 2 * sqrt(1 - v_sq) + self.K
        zero_freq = 0
        if tau == 0:
            zero_freq = self.h / self.T
        zero_freq += self.g * self.K / zn_1
        if fabs(self.freq) <= sqrt(1 - v_sq):
            zn_2 = 2 * sqrt(1 - v_sq - self.freq ** 2) + self.K - self.M * self.freq ** 2
        else:
            zn_2 = -(2 * (self.freq / fabs(self.freq)) * sqrt(
                fabs(1 - v_sq - self.freq ** 2)) - self.K + self.M * self.freq ** 2)
        harmonic_force = self.ampl * cos(self.freq * tau) / zn_2
        return zero_freq + harmonic_force


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
            term_1 = delta #* self.ampl * self.freq * cos(self.phase)
        if tau >= 0:
            term_2 = -self.ampl * (self.freq**2) * sin(self.freq * tau + self.phase) +self.g*self.freq   #?
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


def harmonic_force(syst: SystemUnderStudy, g: float, time: np.ndarray, phase: float = 0., delta_indicator: bool = False,
                   freq: float = 0, amplitude: float = 0) -> np.ndarray:
    delta = float(int(delta_indicator)/(time[1]-time[0]))
    force = np.zeros_like(time, dtype = 'float64')
    max_iter = int(force.shape[0])
    _harmonic_force_(syst, g, phase, delta, time, force, freq, amplitude, max_iter)
    return force


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _harmonic_force_(SystemUnderStudy syst, double g, double phase,
                           double delta, double[:] time, double[:] force, double freq,
                           double amplitude, int iter_max):
    cdef int i
    cdef double term_1, term_2, term_3
    for i in range(iter_max):#, nogil=True):
        force[i] = syst.ext_force(time[i])
        #term_1 = 0
        #if time[i] == 0:
        #    term_1 = delta
        #term_2 = 0
        #term_3 = 0
        #if time[i] >= 0:
        #    term_2 = g
        #    term_3 = amplitude * cos(freq * time[i])# + phase)
        #force[i] =term_1 + term_2 + term_3   #g + amplitude * sin( freq* time[i] + phase)
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
    cdef double term_1, term_2, term_3, freq, time_max, d_tau
    freq = sqrt(-K / M)
    for i in range(1, time.shape[0]):
        time_max = time[i]
        d_tau = time_max / max_iter
        term_1 = integrated_expression_negative(amplitude, bar_omega, K, M, time[i], phase)
        term_2 = g * sinh(freq * time_max) / freq
        term_3 = cosh(freq * time_max)
        rs[i] = 2*(term_1+term_2+term_3)/M


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double integrated_expression_negative(double A, double bar_omega, double K, double M, double t,
                                           double phi)nogil:
    cdef:
        double term_1, term_2, KM, denominator, term_3
    KM = sqrt(-K/M)
    denominator = bar_omega**2+K/M

    term_2 = bar_omega*sin(bar_omega*t)#-bar_omega*cos(KM*t)
    term_3 = KM*sinh(KM*t)

    return A*(term_2+term_3)/denominator
    #return A*(bar_omega*sin(bar_omega*t)+KM*sinh(KM*t))/(bar_omega**2+K/M)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void positive_K(double[:] rs, double[:] time, int max_iter,double amplitude, double bar_omega, double K, double M,
                                  double phase, double g):
    cdef int i
    cdef double term_1, term_2, term_3, freq, time_max, d_tau
    freq = sqrt(abs(K / M))
    for i in range(1, time.shape[0]):
        time_max = time[i]
        d_tau = time_max/max_iter
        term_1 = integrated_expression_positive(amplitude, bar_omega, K, M, time[i], phase)
        term_2 = g * sin(freq * time_max) / freq
        term_3 = cos(freq*time_max)
        rs[i] = (2/M)*(term_1+term_2+term_3)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double integrated_expression_positive(double A, double bar_omega, double K, double M, double t,
                                           double phi)nogil:
    cdef:
        double term_1, term_2, KM, denominator, term_3
    KM = sqrt(K / M)
    denominator = bar_omega ** 2 - K / M

    term_2 = bar_omega * sin(bar_omega * t )
    term_3 = KM * sin(KM * t)

    return A * (term_2 + term_3) / denominator


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
        x_i = integral_0(v,i,t)
        #x_i = integral_c(v[i],v[i-1], v00, t[i], t[i-1])
        for j in range(i+1):
            x_j = integral_0(v,j,t)
            #x_j = integral_c(v[j],v[j-1], v00, t[j], t[j-1])
            J = Bessel(K,M,t[i],t[j],v[i],x_i,x_j)
            out[i,j]=-(J-summand(t[i],t[j],K,M))*h
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
    v_sq = v*v
    O_new(O, v*v, M, K)
    t_len = t.shape[0]
    print(f'O(0)={O[0]}')
    ff = fourier_force(g, O[0], ampl=amplitude, freq=freq, phase=phase, step_time=t[1]-t[0])
    absFp = np.zeros_like(force) + np.abs(ff*np.sqrt(2*pi) ) #  g /O[0]  #
    argFp = np.zeros_like(force) + np.angle(ff*np.sqrt(2*pi) )  #
    #_calc_P(syst, t, return_force, K, M, v*v,force,j, absFp, argFp, U,O, amplitude, freq, phase, t_len)
    _new_calc_P_(syst, t, return_force, np.abs(ff ), np.angle(ff ) , O, v_sq)
    return return_force, absFp, argFp

"""
@cython.boundscheck(False)
@cython.wraparound(False)
cdef (double, double) abs_arg(double complex f)nogil:
    cdef double abs_f, arg_f
    abs_f = 0.
    arg_f = 0.
    abs_f = sqrt(f.real**2+f.imag**2)*sqrt(2*pi)
    arg_f = atan(f.imag/f.real)
    return abs_f, arg_f
"""
"""
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calc_P(SystemUnderStudy syst, double[:] t, double [:] outP0, double K, double M, double[:] v_sq, double[:] p,
                                                                                            int j,
                   double[:] absFp, double[:] argFp, double[:] U,double [:] omega,
                  double amplitude, double freq, double phase, int t_len) nogil:
    cdef:
        int i,l
        double C,integ_omega, C1, C2, omega_0, v_sq_0, C_old, g
        double term_2, term_3, multiplier_1, multiplier_2, absFp_old, argFp_old
        double complex F_p
    l=t_len//50
    g = syst.g
    absFp_old=g/(omega[0])
    argFp_old=pi/2
    C_old = sqrt(
            sqrt(1.-v_sq[0]-omega[0]*omega[0])/
           (omega[0]*( 1.+M*sqrt(1.-v_sq[0]-omega[0]*omega[0]) ))
    )
    omega_0 = omega[0]
    v_sq_0 = v_sq[0]
    C1 = sqrt(
              omega_0 * (1 - v_sq_0 - omega_0**2) * (M*M*omega_0**2 - K*M + 2)
             )
    C2 =  sqrt( M*omega_0**2 - K )*(1 + M * sqrt( 1 - v_sq_0 - omega_0**2)) *omega_0
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
        outP0[i] = step_cicle_for_force(syst, omega[i], K, M, p[i], v_sq[i], absFp[i], argFp[i], C,
                                        integ_omega, t[i], freq, v_sq_0)
    printf('*\n')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double step_cicle_for_force(SystemUnderStudy syst, double omega, double K, double M, double p, double v_sq,
                                 double absFp, double argFp, double C, double integ_omega, double time, double freq,
                                 double v_sq_0)nogil:
    cdef:
        int i
        double term_2, term_3, multiplier_1, multiplier_2, outP0,z_fr_f, p1, zn, sign
    p1 = syst.ext_force(time)
    z_fr_f = _freq_force_new_(syst, time, v_sq, freq, M, K)#*(M*freq**2 - K)
    multiplier_1 = sqrt(
                            (M*omega*omega-K)/
                            (omega*(M*M*omega*omega-K*M+2))
                        )*(M*omega*omega-K)
    multiplier_2 = sin(integ_omega - argFp)
    term_3 = C * multiplier_1 * multiplier_2 * absFp
    outP0 = p1 + term_3 - z_fr_f
    return outP0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _freq_force_new_(SystemUnderStudy syst, double time, double v_sq, double freq,
                             double M, double K)nogil:
    cdef double zero_freq, force_freq, zn_2, p_freq, mult_p_freq
    zero_freq = syst.g*K/(2*sqrt(1-v_sq)+K)
    if fabs(freq)<=sqrt(1-v_sq):
        p_freq = syst.ampl * cos(freq * time)
        zn_2 = 2*sqrt(fabs(1. - v_sq- freq**2))+K-M*freq**2
        mult_p_freq = K - M*freq**2
        force_freq = (p_freq * mult_p_freq)/ zn_2
    else:
        p_freq = syst.ampl * cos(freq * time)
        zn_2 = 2*(freq/fabs(freq))*sqrt(fabs(1. - v_sq - freq**2))- K + M*freq**2
        mult_p_freq = M*freq**2 - K
        force_freq = (p_freq*mult_p_freq)/ zn_2
    return zero_freq+force_freq
"""

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _new_calc_P_(SystemUnderStudy syst, double[:] t, double [:] outP0, double absFp, double argFp,
                        double[:] omega, double[:] v_sq):
    cdef:
        int i
        double C, l, M, K, integ_omega
    l = t.shape[0]//50
    M = syst.M
    K = syst.K
    C = sqrt(sqrt(1-v_sq[0]-omega[0]*omega[0]))*absFp/\
        sqrt(omega[0]+omega[0]*M*sqrt(1-v_sq[0]-omega[0]*omega[0]))
    printf('\nCalculate Analityc\n')
    printf('___________________________________________________\n')
    for i in range(t.shape[0]):#, nogil=True):
        if (i%l)==0:
            printf('*')
        if M!=0:
            integ_omega = integral(omega, i, t[i]-t[i-1])
        else:
            integ_omega = integral_O_M_eq_0(sqrt(v_sq[0]),t[i], sqrt(v_sq[i]),sqrt(v_sq[i-1]), i, t[i+1]-t[i],K)
        outP0[i] = _force_in_dot_(syst, t[i], integ_omega, C, argFp, v_sq[i], omega[i])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _force_in_dot_(SystemUnderStudy syst, double t, double integ_omega, double C, double argFp, double v_sq,
                           double omega):
    cdef:
        double W_0, outP, M, sum_force, K, p
    M = syst.M
    K = syst.K
    p = syst.ext_force(t)
    W_0 = (M*omega*omega - K)*sqrt(
                sqrt(1 - v_sq - omega*omega)/
                (omega + omega*M*sqrt(1 - v_sq - omega*omega))
                )
    sum_force = syst.sum_force_f(t,v_sq)
    outP = p - sum_force +C *W_0*sin(integ_omega - argFp)
    return outP



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex fourier_force(double g, double omega, double ampl, double freq, double phase,
                                  double step_time)nogil:
    cdef:
        double complex p, term_1, term_2, term_3, _sum_
    term_1 = 1.+0j
    term_2 =0. + g*1j/(omega*sqrt(2*pi))
    if omega == 0.:
        printf('Omega equal 0')
        term_2 += 1/step_time + 0j
        #term_2 += (1/step_time)*g + 0j
    term_3 = ampl*(1j*freq)/(freq**2-omega**2)
    #term_3 = ampl*(1j*freq)/(omega**2-freq**2)
    if freq == omega:
        printf('Omega equal frequency force')
        #term_3 += 0.5*sqrt(pi/2)*ampl/step_time
        term_3 += 1/step_time
    if freq == -omega:
        printf('Omega equal -1*frequency force')
        term_3 += 1 / step_time
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





def calc_U_analityc(t: np.ndarray, K: float, M: float, v: np.ndarray,
                    force: np.ndarray, j: int, freq: float,
                    amplitude: float, phase: float, g: float, syst: SystemUnderStudy) -> np.ndarray:
    U = np.zeros_like(t, dtype=np.float64)
    O = np.zeros_like(t, dtype='float64')
    v_sq = v*v
    O_new(O, v_sq, M, K)
    ff = fourier_force(g, O[0], ampl=amplitude, freq=freq, phase=phase, step_time=t[1] - t[0])
    absFp = np.abs(ff)  #
    argFp = np.angle(ff )
    _calc_U_analityc_(U, O, v_sq, M, K, absFp, argFp,  t.shape[0], t, force, freq, g, syst)
    print(f'U_a = {U}')
    return U



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calc_U_analityc_(double[:] U, double [:] omega, double[:] v_sq, double M, double K, double absFp,
                            double argFp, int t_len, double[:] t, double[:] p, double freq, double g,
                            SystemUnderStudy syst):
    pass
    cdef:
        int i
    C = sqrt(sqrt(1. - v_sq[0] - omega[0] * omega[0]) / (
                    omega[0] * (1. + M * sqrt(1. - v_sq[0] - omega[0] * omega[0]))))
    C = C*absFp#/omega[0]
    for i in range(t_len):
        if M!=0:
            integ=integral(omega, i, t[i]-t[i-1])
        else:
            integ=integral_O_M_eq_0(sqrt(v_sq[0]),t[i], sqrt(v_sq[i]),sqrt(v_sq[i-1]), i, t[i+1]-t[i],K)
        U[i] = _step_time_U_(p[i], v_sq[i], K, M, omega[i], integ, argFp, C, freq, t[i], g, syst)



cdef double _step_time_U_(double p, double v_sq, double K, double M, double omega, double integ, double argFp,
                          double C, double freq, double t, double g, SystemUnderStudy syst):
    cdef double term_2, multiplier_1, multiplier_2
    cdef double complex term_1
    #term_1 = p/(2*sqrt(fabs(1-v_sq-syst.freq**2))-syst.K+syst.M*syst.freq**2)#-1/(2*sqrt(fabs(1-v_sq-freq**2))+K-M*freq*freq))*cos(freq*t)/2
    term_1 = 0.
    if fabs(syst.freq)<=sqrt(1-v_sq):
        term_1 = syst.ampl*np.exp(-1j*syst.freq*t)/(2*sqrt(fabs(1-v_sq-syst.freq**2))+syst.K-syst.M*syst.freq**2)#(syst.ext_force(t)-g) #(p-syst.g)
    else:
        term_1 = -syst.ampl*np.exp(-1j*syst.freq*t)/(2j*sqrt(-1+v_sq+syst.freq**2)-syst.K+syst.M*syst.freq**2)#(syst.ext_force(t)-g) #(p-syst.g)
    term_1 += syst.g/(2*sqrt(1-v_sq)+syst.K)
    #term_1 = p/(2*sqrt(1-v_sq)+syst.K)
    multiplier_1 = sqrt(
        sqrt(1-v_sq-omega**2)/
        (omega*(1+M*sqrt(1-v_sq-omega**2)))
    )
    multiplier_2 = sin(integ - argFp)
    term_2 = C*multiplier_1*multiplier_2
    return  term_1.real + term_2 #


#Use this function, if external force p=const for all t
def calc_U(t,K,M,P,p, v):
    out_U=np.zeros_like(P,dtype='float64')
    _calc_U_(out_U,t,K,M,P,p, t.shape[0], v)
    print(f'U = {out_U}')
    return out_U

# TODO: Off c_module!!!
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calc_U_(double [:] U, long double [:]t, double K,double M,double [:] P, double[:] p, int len_t,
                   long double [:] v)nogil:
    cdef int i,j
    cdef double T,h,f1,f2,l, sqrt_KM, KM, J1, J2, ko, x_i, x_j
    if M!=0:
        sqrt_KM = sqrt(fabs(K/M))
    else:
        sqrt_KM = 0
    KM = sqrt(fabs(K)*M)
    T=t[len_t-1]
    h=t[1]-t[0]
    l=len_t//50
    printf('\n Calculate moving: \n')
    printf('___________________________________________________\n')
    if (M > 0) & (K >= 0):
        for i in prange(2,len_t, nogil=True):
            if (i % l) == 0:
                printf('#')
            #x_i = integral_c(v[i], v[i - 1], v[0], t[i], t[i - 1])
            x_i = integral_0(v, i, t)
            for j in range(1, i):
                x_j = integral_0(v, j, t)
                ko = (t[i] - t[j]) ** 2 - (x_i - x_j) ** 2
                J1 = P[j] * csc.j0(sqrt(ko)) * h / 4
                x_j = integral_0(v, j - 1, t)
                #x_j = integral_c(v[j], v[j - 1], v[0], t[j], t[j - 1])
                ko = (t[i] - t[j - 1]) ** 2 - (x_i - x_j) ** 2
                J2 = P[j - 1] * csc.j0(sqrt(ko)) * h / 4
                U[i] += J1 + J2
            #f2 = (p[0] - P[0]) * sin(sqrt_KM * (t[i] - t[0]))
            #for j in range(1,i):
            #    f1 = (p[j] - P[j]) * sin(sqrt_KM * (t[i] - t[j]))
            #    #f2 = (p[j - 1] - P[j - 1]) * sin(sqrt_KM * (t[i] - t[j - 1]))
            #    U[i] += (f1 + f2) * h / (2 * KM)
            #    f2 = f1
    elif (M > 0) & (K < 0):
        for i in prange(2,len_t, nogil=True):
            if (i % l) == 0:
                printf('#')
            x_i = integral_0(v, i, t)
            #x_i = integral_c(v[i],v[i-1], v[0], t[i], t[i-1])
            for j in range(1,i):
                x_j = integral_0(v, j, t)
                ko = (t[i] - t[j]) ** 2 - (x_i - x_j) ** 2
                J1 = P[j] * csc.j0(sqrt(ko)) * h / 4
                x_j = integral_0(v, j - 1, t)
                #x_j = integral_c(v[j],v[j-1], v[0], t[j], t[j-1])
                ko = (t[i] - t[j - 1]) ** 2 - (x_i - x_j) ** 2
                J2 = P[j - 1] * csc.j0(sqrt(ko)) * h / 4
                U[i] += J1 + J2

            #f2 = (p[0] - P[0]) * sinh(sqrt_KM * (t[i] - t[0]))
            #for j in range(1,i):
            #    f1 = (p[j] - P[j]) * sinh(sqrt_KM * (t[i] - t[j]))
            #   # f2 = (p[j - 1] - P[j - 1]) * sinh(sqrt_KM * (t[i] - t[j - 1]))
            #    U[i] += (f1 + f2) * h / (2 * KM)
            #    f2=f1
    elif (M==0)&(K<0):
        for i in prange(0,len_t, nogil=True):
            if (i%l)==0:
                printf('#')
            U[i]+=(P[i]-p[i])/(-K)
    else:
        printf('\n Error! Failed parametrs! Func _calc_U: \n K= %f, M=%f ; \n', K, M)
    printf('##\n')


'''    
def calc_p2(t,K1,M,g):
    out = np.zeros_like(t, dtype='float64')
    _calc_p2(t, out, K1, M,g)
    return out
'''




