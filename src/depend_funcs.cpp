// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
arma::vec prox_func_cpp(arma::vec y, double C){
  double y_norm2 = arma::accu(square(y));
  if(y_norm2 <= C){
    return y;
  }
  else{
    return sqrt(C / y_norm2) * y;
  }
}
arma::vec prox_func_theta_cpp(arma::vec y, double C){
  double y_norm2 = arma::accu(square(y)) - 1;
  if(y_norm2 <= C){
    return y;
  }
  else{
    y = sqrt(C / y_norm2) * y;
    y(0) = 1;
    return y;
  }
}
