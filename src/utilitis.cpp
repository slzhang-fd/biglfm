#include <RcppArmadillo.h>
#include "biglfm_omp.h"

//' @export
// [[Rcpp::export]]
double myKendall_cpp(arma::vec x, arma::vec y){
  double res = 0;
  int n = x.n_elem;
  double nn = x.n_elem;
#pragma omp parallel for reduction(+:res)
  for(int i=0;i<n-1;++i){
    for(int j=i+1;j<n;++j){
      if( (x(i) - x(j))*(y(i) - y(j)) < 0 )
        res += 1;
    }
  }
  res = res * 2.0 / (nn * (nn-1)) ;
//  Rprintf("res= %f\n", res);
  return res;
}
double myKendall_matrix_max(arma::mat x, arma::mat y){
  arma::vec tmp(x.n_cols);
  for(unsigned int i=0;i<x.n_cols;++i){
    tmp(i) = std::min(myKendall_cpp(x.col(i), y.col(i)), myKendall_cpp(x.col(i), -y.col(i)));
  }
  return tmp.max();
}
arma::mat sample_A_func_old(int K, int J, double C, double frac){
  arma::mat A(K,J);
  double radis = std::sqrt(C);
  for(unsigned int i=0;i<J;++i){
    arma::vec tmp = radis * arma::randu<arma::vec>(K);
    while(arma::accu(arma::square(tmp)) > C ||
          arma::accu(arma::square(tmp)) < C * frac){
      tmp = radis * arma::randu<arma::vec>(K);
    }
    A.col(i) = tmp;
  }
  return A.t();
}
//' @export
// [[Rcpp::export]]
arma::mat sample_A_func(int K, int J, double C, double frac){
  arma::mat A(K,J);
  double radis = std::sqrt(C);
  arma::vec rr = radis * std::sqrt(frac) + radis *(1-std::sqrt(frac)) * arma::pow(arma::randu<arma::vec>(J), 1.0/K);
  for(unsigned int i=0;i<J;++i){
    arma::vec dir = arma::randn(K);
    dir = arma::abs(dir) / std::sqrt(arma::accu(arma::square(dir)));
    A.col(i) = dir * rr(i);
  }
  return A.t();
}
arma::mat sample_theta_func_old(int K, int N, double C, double frac){
  arma::mat theta(K,N);
  double radis = std::sqrt(C);
  for(unsigned int i=0;i<N;++i){
    arma::vec tmp = radis * (2*arma::randu<arma::vec>(K) - 1);
    while(arma::accu(arma::square(tmp)) > C ||
          arma::accu(arma::square(tmp)) < C * frac){
      tmp = radis * (2*arma::randu<arma::vec>(K) - 1);
    }
    theta.col(i) = tmp;
  }
  return theta.t();
}
//' @export
// [[Rcpp::export]]
arma::mat sample_theta_func(int K, int N, double C, double frac){
  arma::mat theta(K,N);
  double radis = std::sqrt(C);
  arma::vec rr = radis * std::sqrt(frac) + radis*(1-std::sqrt(frac)) * arma::pow(arma::randu<arma::vec>(N), 1.0/K);
  for(unsigned int i=0;i<N;++i){
    arma::vec dir = arma::randn(K);
    dir = dir / std::sqrt(arma::accu(arma::square(dir)));
    theta.col(i) = dir * rr(i);
  }
  return theta.t();
}
