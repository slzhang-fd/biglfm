#ifndef __DEPEND_FUNCS__
#define __DEPEND_FUNCS__
#include <RcppArmadillo.h>

arma::vec prox_func_cpp(arma::vec y, double C);
arma::vec prox_func_theta_cpp(arma::vec y, double C);
#endif

