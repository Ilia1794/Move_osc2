from libc.math cimport sqrt, pow, exp,cos,sin, cosh, sinh, fabs,pi, atan, log
cimport cython
from cython.parallel cimport prange
from libc.stdio cimport printf
import tqdm
import numpy as np
import scipy.special as sc
import scipy
cimport scipy.special.cython_special as csc


def hello_world():
    _hello_world_()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _hello_world_()nogil:
    printf('hello_world')


def harmonic_force(g: float, time: np.ndarray,
                   phase: float = 0., delta_indicator: bool = False) -> np.ndarray:
    delta = float(int(delta_indicator)/(time[1]-time[0]))
    force = np.zeros_like(time, dtype = 'float64')
    max_iter = int(force.shape[0])
    _harmonic_force_(g, phase, delta, time, force, max_iter)
    return force


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _harmonic_force_(double g, double phase,
                           double delta, double[:] time, double[:] force, int iter_max)nogil:
    cdef int i
    for i in prange(iter_max, nogil=True):
        force[i] = g + amplitude_force(time[i]) * sin(frequency_force(time[i])* time[i] + phase)
    force[0] = delta


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double amplitude_force(double time)nogil:
    return 1#log(fabs(time) + 1.5)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double frequency_force(double time)nogil:
    return 5 #* cos(time / 100)


def right_side_for_harmonic_force(g, phase,time,K,M):
    max_iter = 10000
    freq = sqrt(K / M)
    print(f'freq= {freq}')
    rs = np.zeros_like(time)
    for i in tqdm.tqdm(range(1, time.shape[0])):
        time_max = time[i]
        tau = np.linspace(0,time_max, max_iter)
        intag = np.zeros(max_iter, dtype='float64')
        #rs[i] = scipy.integrate.quad(lambda x :_integrand_(g, time_max, freq, phase, x), 0.,
        #                            time_max)[0]
        rs[i]=_right_side_for_harmonic_force_(g, phase, time_max, tau,max_iter, freq, intag)
        rs[i] += g * sin(freq * time_max) / freq
        rs[i]+= cos(freq*time_max)
        rs[i] *= 2/M
    return rs



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _right_side_for_harmonic_force_(double g, double phase, double time, double[:] tau,
                                          int max_iter, double freq, double[:] rs)nogil:
    cdef:
        int i, j
        double step_1, step_2,  step_time, ret
    #rs = 0
    step_1 = 0.
    step_2 = 0.
    step_time = tau[1]-tau[0]
    #step_2 = _integrand_(g, time, freq, phase, tau[0])
    for i in prange(1, max_iter, nogil=True):
        step_2 = _integrand_(g, time, freq, phase, tau[i-1])
        step_1 = _integrand_(g, time, freq, phase, tau[i])
        step_time = tau[i] - tau[i-1]
        rs[i] =(step_1+step_2)*step_time/2
        step_2 = 0.#step_1
        step_1 = 0.
    step_1 = 0.
    step_2 = 0.
    ret = 0
    i=1
    for i in prange(1, max_iter, nogil=True):
        ret += rs[i]
    return ret#rs


def integrand(g, time, freq, phase, tau):
    _integrand_(g, time, freq, phase, tau)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _integrand_(double g, double time_max, double freq_cos, double phase, double tau)nogil:
    cdef double step
    step = 0.
    #step = g + amplitude_force(tau) * sin(frequency_force(tau)* tau + phase)
    step = amplitude_force(tau) * sin(frequency_force(tau)* tau + phase)
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
            Omega[i]=1-v[i]*v[i]-K*K/4


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
    _parallel_calc_matrices_1(t, out, K, M, v, h)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _parallel_calc_matrices_1(long double[:] t, long double[:,:] out, long double K,
                                  long double M, long double[:] v, long double h) nogil:
    cdef  int i, j, l
    cdef long double x_i, x_j,J
    j=0
    i=0
    l=len(t)//50
    printf('\n Calculate matrices: \n')
    printf('___________________________________________________\n')
    for i in prange(t.shape[0], nogil=True):
        if (i%l)==0:
            printf('#')
        x_i=integral_0(v,i,t)
        for j in range(i+1):
            x_j=integral_0(v,j,t)
            J=Bessel(K,M,t[i],t[j],v[i],x_i,x_j)
            out[i,j]=-(J-summand(t[i],t[j],K,M))*h
    printf('#\n')
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
    four_force= np.fft.fft(force)
    absFp = np.abs(four_force)
    argFp = np.angle(four_force)
    _calc_P1(t, outP0, K1, M, v,force,j,absFp,argFp, U,O)

    return outP0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calc_P1(double[:] t, double [:] outP0, double K1, double M, double[:] v, double[:] g,
                                                                                            int j,
                   double[:] absFp, double[:] argFp, double[:] U,double [:] omega) nogil:
    cdef int i,l
    cdef double C,integ,O0
    l=len(t)//50
    O(omega,v,M,K1)
    #absFp=g/(omega[0])
    #argFp=pi/2
    C=sqrt(sqrt(1.-v[0]*v[0]-omega[0]*omega[0])/(omega[0]*(1.+M*sqrt(1.-v[0]**2-omega[0]*omega[0]) )))
    printf('\n Calculate Analityc, C= %f\n',C);    printf('___________________________________________________\n')
    i=0
    for i in prange(t.shape[0]):
        if (i%l)==0:
            printf('*')
        if M!=0:
            integ=integral(omega, i, t[i]-t[i-1])
        else:
            integ=integral_O_M_eq_0(v[0],t[i],v[i],v[i-1],i,t[i+1]-t[i],K1)
        O0=K1*g[i]/(2.*sqrt(1.-v[i]**2)+K1)
        U[i]=C*sqrt(sqrt(1.-v[i]*v[i]-omega[i]*omega[i])/
                    (omega[i]*(1.+M*sqrt(1.-v[i]*v[i]-omega[i]*omega[i]))))*(M*omega[i]*omega[i]-K1)
        outP0[i]=g[i]-O0+U[i]*absFp[i]*sin(integ-argFp[i])

    printf('*\n')


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
    printf('\n Calculate Analityc \n');    printf('___________________________________________________\n')
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







