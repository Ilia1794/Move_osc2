//
// Created by USER on 12.08.2021.
//

#include "func.h"
#include "math.h"
#include <stdio.h>


long double Bessel_c(double K, double M, long double ti, long double tj, long double v,
                     long double x_i,long double x_j){
    long double Be,ko, s_ko;
    Be = 0;
    ko=(ti-tj)*(ti-tj)-(x_i-x_j)*(x_i-x_j);
    s_ko = sqrtl(ko);
    if ((ko>0.)&(M>0.)){
        Be=(ti-tj-(x_i-x_j)*v)*_j1(s_ko)/s_ko;
    }
    else if ((ko==0)&(M>0)){
        Be=(ti-tj-(x_i-x_j)*v);
    }
    else if ((ko==0.)&(M==0.)&(K<0)){
        Be=fabs(K)/2.;
    }
    else if ((ko>=0.)&(M==0.)&(K<0)){
        Be=(-K)*_j0(s_ko)/2;
    }
    else if ((ko<0.)&(M>=0.)){
        Be=0.;
    }
    else{
        printf("Error! Failed parametrs! Func Bessel");
        Be = 0.;
    }
    return Be;
}

long double summand_c(long double ti,long double tj, long double K, long double M){
    long double S;
    S = 0.;
    if ((K>0.)&(M>0)){
        S=2.*cosl((ti-tj)*sqrt(K/M))/M;
    }
    else if ((K<0.)&(M>0)){
        S=2.*coshl((ti-tj)*sqrt(fabs(K/M)))/M;
    }
    else if ((K==0.)&(M>0)){
        S=2./M;
    }
    else if ((K<0)&(M==0)){
        S = 0.;
    }
    return S;
}

long double integral_c(long double v_1, long double v_0, long double v00 , long double  t_1, long double  t_0){
    long double s;
    s = v00 * t_1;
    s += t_1*t_1*((v_1-v_0)/(t_1-t_0))/2;
    return s;
}

