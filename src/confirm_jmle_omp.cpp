#include <RcppArmadillo.h>
#include "biglfm_omp.h"
#include "depend_funcs.h"

using namespace std;
// [[Rcpp::depends(RcppArmadillo)]]
double neg_loglik(const arma::mat &thetaA, const arma::mat &response){
  double res = arma::accu( thetaA % response - log(1+exp(thetaA)) );
  return -res;
}
double neg_loglik_i_cpp(const arma::vec &response_i, const arma::vec &nonmis_ind_i,
                        const arma::mat &A, const arma::vec &theta_i){
  arma::vec tmp = A * theta_i;
  return -arma::accu(nonmis_ind_i % (tmp % response_i - log(1 + exp(tmp))));
}
arma::vec grad_neg_loglik_thetai_cpp(const arma::vec &response_i, const arma::vec &nonmis_ind_i,
                                     const arma::mat &A, const arma::vec &theta_i){
  arma::vec tmp = response_i - 1 / (1 + exp(- A * theta_i));
  arma::mat tmp1 = -arma::diagmat(nonmis_ind_i % tmp) * A;
  return tmp1.t() * arma::ones(response_i.n_rows);
}

// [[Rcpp::plugins(openmp)]]
arma::mat Update_theta_cpp(const arma::mat &theta0, const arma::mat &response, const arma::mat &nonmis_ind,
                           const arma::mat &A0, double C){
  arma::mat theta1 = theta0.t();
  int N = response.n_rows;
#pragma omp parallel for
  for(int i=0;i<N;++i){
    arma::vec theta0_i_tmp = theta0.row(i).t();
    arma::vec theta1_i_tmp = theta1.col(i);
    double step = 10;
    arma::vec h = grad_neg_loglik_thetai_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0_i_tmp);
    h(0) = 0;
    theta1_i_tmp = theta0_i_tmp - step * h;
    theta1_i_tmp = prox_func_theta_cpp(theta1_i_tmp, C);
    while(neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta1_i_tmp) >
            neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0_i_tmp) &&
          step > 1e-7){
      step *= 0.5;
      theta1_i_tmp = theta0_i_tmp - step * h;
      theta1_i_tmp = prox_func_theta_cpp(theta1_i_tmp, C);
    }
    if(step <= 1e-7){
      Rprintf("warning, threshould of step size of theta needs to be smaller. It's likely the tol setting is too small.\n");
    }
    while(neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0_i_tmp)-
          neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta1_i_tmp) > 100){
      theta0_i_tmp = theta1_i_tmp;
      h = grad_neg_loglik_thetai_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0_i_tmp);
      h(0) = 0;
      theta1_i_tmp = theta0_i_tmp - step * h;
      theta1_i_tmp = prox_func_theta_cpp(theta1_i_tmp, C);
      while(neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta1_i_tmp) >
              neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0_i_tmp) &&
              step > 1e-7){
        step *= 0.5;
        theta1_i_tmp = theta0_i_tmp - step * h;
        theta1_i_tmp = prox_func_theta_cpp(theta1_i_tmp, C);
      }
    }
    theta1.col(i) = theta1_i_tmp;
  }
  return(theta1.t());
}

arma::mat Update_theta_nob_cpp(const arma::mat &theta0, const arma::mat &response, const arma::mat &nonmis_ind,
                               const arma::mat &A0, double C){
  arma::mat theta1 = theta0.t();
  int N = response.n_rows;
#pragma omp parallel for
  for(int i=0;i<N;++i){
    arma::vec theta0_i_tmp = theta0.row(i).t();
    arma::vec theta1_i_tmp = theta1.col(i);
    double step = 10;
    arma::vec h = grad_neg_loglik_thetai_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0_i_tmp);
    theta1_i_tmp = theta0_i_tmp - step * h;
    theta1_i_tmp = prox_func_cpp(theta1_i_tmp, C);
    while(neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta1_i_tmp) >
            neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0_i_tmp) &&
            step > 1e-7){
      step *= 0.5;
      theta1_i_tmp = theta0_i_tmp - step * h;
      theta1_i_tmp = prox_func_cpp(theta1_i_tmp, C);
    }
    if(step <= 1e-7){
      Rprintf("warning, threshould of step size of theta needs to be smaller. It's likely the tol setting is too small.\n");
    }
    while(neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0_i_tmp)-
          neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta1_i_tmp) > 100){
      theta0_i_tmp = theta1_i_tmp;
      h = grad_neg_loglik_thetai_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0_i_tmp);
      theta1_i_tmp = theta0_i_tmp - step * h;
      theta1_i_tmp = prox_func_cpp(theta1_i_tmp, C);
      while(neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta1_i_tmp) >
              neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0_i_tmp) &&
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

double neg_loglik_j_cpp(const arma::vec &response_j, const arma::vec &nonmis_ind_j, const arma::vec &A_j,
                        const arma::mat &theta){
  arma::vec tmp = theta * A_j;
  return -arma::accu(nonmis_ind_j % (tmp % response_j - log(1+exp(tmp))));
}

arma::vec grad_neg_loglik_A_j_cpp(const arma::vec &response_j, const arma::vec &nonmis_ind_j,
                                  const arma::vec &A_j, const arma::vec &Q_j, const arma::mat &theta){
  arma::vec tmp = response_j - 1 / (1 + exp(-theta * A_j));
  arma::vec tmp1 = nonmis_ind_j % tmp;
  arma::vec res = theta.row(0).t() * tmp1(0);
  for(int i=1;i<theta.n_rows;++i){
    res += theta.row(i).t() * tmp1(i);
  }
  return -res % Q_j;
}
// [[Rcpp::plugins(openmp)]]
arma::mat Update_A_cpp(const arma::mat &A0, const arma::mat &Q, const arma::mat &response,
                       const arma::mat &nonmis_ind, const arma::mat &theta1, double C){
  arma::mat A1 = A0.t();
  int J = A0.n_rows;
#pragma omp parallel for
  for(int j=0;j<J;++j){
    double step = 5;
    arma::vec A0_j_tmp = A0.row(j).t();
    arma::vec A1_j_tmp = A1.col(j);
    arma::vec h = grad_neg_loglik_A_j_cpp(response.col(j), nonmis_ind.col(j), A0_j_tmp, Q.row(j).t(), theta1);
    A1_j_tmp = A0_j_tmp - step * h;
    A1_j_tmp = prox_func_cpp(A1_j_tmp, C);
    while(neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A1_j_tmp, theta1) >
            neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A0_j_tmp, theta1) &&
          step > 1e-7){
      step *= 0.5;
      A1_j_tmp = A0_j_tmp - step * h;
      A1_j_tmp = prox_func_cpp(A1_j_tmp,C);
    }
    if(step <= 1e-7){
      Rprintf("warning, threshould of step size of A needs to be smaller. It's likely the tol setting is too small.\n");
    }
    while(neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A0_j_tmp, theta1) -
            neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A1_j_tmp, theta1) > 100){
      A0_j_tmp = A1_j_tmp;
      h = grad_neg_loglik_A_j_cpp(response.col(j), nonmis_ind.col(j), A0_j_tmp, Q.row(j).t(), theta1);
      A1_j_tmp = A0_j_tmp - step * h;
      A1_j_tmp = prox_func_cpp(A1_j_tmp, C);
      while(neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A1_j_tmp, theta1) >
              neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A0_j_tmp, theta1) &&
              step > 1e-7){
        step *= 0.5;
        A1_j_tmp = A0_j_tmp - step * h;
        A1_j_tmp = prox_func_cpp(A1_j_tmp,C);
      }
    }
    A1.col(j) = A1_j_tmp;
  }
  return(A1.t());
}
//' @export
// [[Rcpp::export]]
Rcpp::List confirm_CJMLE_MIRTb_cpp( arma::mat response,  arma::mat nonmis_ind, arma::mat theta0,
                                    arma::mat A0, arma::mat Q, double C, double tol){
  int K = theta0.n_cols;
  arma::mat theta1 = Update_theta_cpp(theta0, response, nonmis_ind, A0, C);
  arma::mat A1 = Update_A_cpp(A0, Q, response, nonmis_ind, theta1, C);
  double eps = neg_loglik(theta0*A0.t(), response) - neg_loglik(theta1*A1.t(), response);
  while(eps > tol){
    Rprintf("eps: %f\n",eps);
    theta0 = theta1;
    A0 = A1;
    theta1 = Update_theta_cpp(theta0, response, nonmis_ind, A0, C);
    A1 = Update_A_cpp(A0, Q, response, nonmis_ind, theta1, C);
    eps = neg_loglik(theta0*A0.t(), response) - neg_loglik(theta1*A1.t(), response);
  }
  return Rcpp::List::create(Rcpp::Named("A") = A1.cols(1,K-1),
                            Rcpp::Named("b") = A1.col(0),
                            Rcpp::Named("theta") = theta1.cols(1,K-1),
                            Rcpp::Named("obj") = neg_loglik(theta1*A1.t(), response));
}
//' @export
// [[Rcpp::export]]
Rcpp::List CJMLE_MIRT(const arma::mat &response, const arma::mat &nonmis_ind,
                                           arma::mat theta0, arma::mat A0, const arma::mat &Q,
                                           double C, double tol, bool parallel){
  if(!parallel)
    omp_set_num_threads(1);
  arma::mat theta1 = Update_theta_nob_cpp(theta0, response, nonmis_ind, A0, C);
  arma::mat A1 = Update_A_cpp(A0, Q, response, nonmis_ind, theta1, C);
  double eps = neg_loglik(theta0*A0.t(), response) - neg_loglik(theta1*A1.t(), response);
  while(eps > tol){
    Rprintf("eps: %f\n", eps);
    Rcpp::checkUserInterrupt();
    theta0 = theta1;
    A0 = A1;
    theta1 = Update_theta_nob_cpp(theta0, response, nonmis_ind, A0, C);
    A1 = Update_A_cpp(A0, Q, response, nonmis_ind, theta1, C);
    eps = neg_loglik(theta0*A0.t(), response) - neg_loglik(theta1*A1.t(), response);
  }
  return Rcpp::List::create(Rcpp::Named("A") = A1,
                            Rcpp::Named("theta") = theta1,
                            Rcpp::Named("obj") = neg_loglik(theta1*A1.t(), response));
}
