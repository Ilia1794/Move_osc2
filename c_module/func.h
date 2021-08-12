//
// Created by USER on 12.08.2021.
//

#ifndef C_MODULE_FUNC_H
#define C_MODULE_FUNC_H


long double Bessel_c(double K, double M, long double ti, long double tj, long double v,
                     long double x_i,long double x_j);
long double summand_c(long double ti,long double tj, long double K, long double M);
long double integral_c(long double v_1, long double v_0, long double v00 , long double  t_1, long double  t_0);


#endif //C_MODULE_FUNC_H
