#include <RcppArmadillo.h>
#include "biglfm_omp.h"
#include "depend_funcs.h"

// [[Rcpp::depends(RcppArmadillo)]]

double neg_loglik_poisson(const arma::mat &thetaA, const arma::mat &response){
  return arma::accu(arma::exp(thetaA) - response % thetaA);
}

double neg_loglik_poisson_i_cpp(const arma::vec &response_i, const arma::mat &A, const arma::vec &theta_i){
  arma::vec tmp = A * theta_i;
  return arma::accu(arma::exp(tmp) - response_i % tmp);
}

arma::vec grad_neg_loglik_poisson_thetai_cpp(const arma::vec &response_i, const arma::mat &A, const arma::vec &theta_i){
  arma::vec tmp = A * theta_i;
  return A.t() * (arma::exp(tmp) - response_i);
}

arma::mat Update_theta_poisson_cpp(const arma::mat &theta0, const arma::mat &response, const arma::mat &A0, double C){
  arma::mat theta1 = theta0.t();
  int N = response.n_rows;
#pragma omp parallel for
  for(int i=0;i<N;++i){
    double step = 10;
    arma::vec theta0_i_tmp = theta0.row(i).t();
    arma::vec theta1_i_tmp = theta1.col(i);
    arma::vec h = grad_neg_loglik_poisson_thetai_cpp(response.row(i).t(), A0, theta0_i_tmp);
    theta1_i_tmp = theta0_i_tmp - step * h;
    theta1_i_tmp = prox_func_cpp(theta1_i_tmp, C);
    while(neg_loglik_poisson_i_cpp(response.row(i).t(), A0, theta1_i_tmp) >
            neg_loglik_poisson_i_cpp(response.row(i).t(), A0, theta0_i_tmp) &&
          step > 1e-7){
      step *= 0.5;
      theta1_i_tmp = theta0_i_tmp - step * h;
      theta1_i_tmp = prox_func_cpp(theta1_i_tmp, C);
    }
    if(step <= 1e-7){
      Rprintf("warning, threshould of step size of theta needs to be smaller. It's likely the tol setting is too small.\n");
    }
    while(neg_loglik_poisson_i_cpp(response.row(i).t(), A0, theta0_i_tmp) -
            neg_loglik_poisson_i_cpp(response.row(i).t(), A0, theta1_i_tmp) > 100){
      theta0_i_tmp = theta1_i_tmp;
      h = grad_neg_loglik_poisson_thetai_cpp(response.row(i).t(), A0, theta0_i_tmp);
      theta1_i_tmp = theta0_i_tmp - step * h;
      theta1_i_tmp = prox_func_cpp(theta1_i_tmp, C);
      while(neg_loglik_poisson_i_cpp(response.row(i).t(), A0, theta1_i_tmp) >
              neg_loglik_poisson_i_cpp(response.row(i).t(), A0, theta0_i_tmp) &&
              step > 1e-7){
        step *= 0.5;
        theta1_i_tmp = theta0_i_tmp - step * h;
        theta1_i_tmp = prox_func_cpp(theta1_i_tmp, C);
      }
    }
    theta1.col(i) = theta1_i_tmp;
  }
  return(theta1.t());
}

double neg_loglik_poisson_j_cpp(const arma::vec &response_j, const arma::vec &A_j, const arma::mat &theta){
  arma::vec tmp = theta * A_j;
  return arma::accu(arma::exp(tmp) - response_j % tmp);
}

arma::vec grad_neg_loglik_poisson_A_j_cpp(const arma::vec &response_j, const arma::vec &A_j, const arma::vec &Q_j,
                                          const arma::mat &theta){
  arma::vec tmp = theta * A_j;
  return (theta.t() * (arma::exp(tmp) - response_j)) % Q_j;
}
arma::mat Update_A_poisson_cpp(const arma::mat &A0, const arma::mat &Q, const arma::mat &response,
                               const arma::mat &theta1, double C){
  arma::mat A1 = A0.t();
  int J = A0.n_rows;
#pragma omp parallel for
  for(int j=0;j<J;++j){
    double step = 5;
    arma::vec A0_j_tmp = A0.row(j).t();
    arma::vec A1_j_tmp = A1.col(j);
    arma::vec h = grad_neg_loglik_poisson_A_j_cpp(response.col(j), A0_j_tmp, Q.row(j).t(), theta1);
    A1_j_tmp = A0_j_tmp - step * h;
    A1_j_tmp = prox_func_cpp(A1_j_tmp, C);
    while(neg_loglik_poisson_j_cpp(response.col(j), A1_j_tmp, theta1) >
            neg_loglik_poisson_j_cpp(response.col(j), A0_j_tmp, theta1) &&
          step > 1e-7){
      step *= 0.5;
      A1_j_tmp = A0_j_tmp - step * h;
      A1_j_tmp = prox_func_cpp(A1_j_tmp, C);
    }
    if(step <= 1e-7){
      Rprintf("warning, threshould of step size of A needs to be smaller. It's likely the tol setting is too small.\n");
    }
    while(neg_loglik_poisson_j_cpp(response.col(j), A0_j_tmp, theta1) -
            neg_loglik_poisson_j_cpp(response.col(j), A1_j_tmp, theta1) > 100){
      A0_j_tmp = A1_j_tmp;
      h = grad_neg_loglik_poisson_A_j_cpp(response.col(j), A0_j_tmp, Q.row(j).t(), theta1);
      A1_j_tmp = A0_j_tmp - step * h;
      A1_j_tmp = prox_func_cpp(A1_j_tmp, C);
      while(neg_loglik_poisson_j_cpp(response.col(j), A1_j_tmp, theta1) >
              neg_loglik_poisson_j_cpp(response.col(j), A0_j_tmp, theta1) &&
              step > 1e-7){
        step *= 0.5;
        A1_j_tmp = A0_j_tmp - step * h;
        A1_j_tmp = prox_func_cpp(A1_j_tmp, C);
      }
    }
    A1.col(j) = A1_j_tmp;
  }
  return(A1.t());
}
//' @export
// [[Rcpp::export]]
Rcpp::List CJMLE_poisson(const arma::mat &response, arma::mat theta0, arma::mat A0, const arma::mat &Q,
                                     double C, double tol, bool parallel){
  arma::mat theta1 = Update_theta_poisson_cpp(theta0, response, A0, C);
  arma::mat A1 = Update_A_poisson_cpp(A0, Q, response, theta1, C);
  double eps = neg_loglik_poisson(theta0*A0.t(), response) - neg_loglik_poisson(theta1*A1.t(), response);
  while(eps > tol){
    Rprintf("eps: %f\n",eps);
    theta0 = theta1;
    A0 = A1;
    theta1 = Update_theta_poisson_cpp(theta0, response, A0, C);
    A1 = Update_A_poisson_cpp(A0, Q, response, theta1, C);
    eps = neg_loglik_poisson(theta0*A0.t(), response) - neg_loglik_poisson(theta1*A1.t(), response);
  }
  return Rcpp::List::create(Rcpp::Named("A") = A1,
                            Rcpp::Named("theta") = theta1,
                            Rcpp::Named("obj") = neg_loglik_poisson(theta1*A1.t(), response));
}

