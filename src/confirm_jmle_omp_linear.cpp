#include <RcppArmadillo.h>
#include "biglfm_omp.h"
#include "depend_funcs.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]


//' @export
// [[Rcpp::export]]
double neg_loglik_linear(const arma::mat &thetaA, const arma::mat &response, double sigma_square){
  //return log(sigma_square) + 1.0 / J / N / sigma_square * arma::accu(arma::square(response - thetaA));
  return arma::accu(arma::square(response - thetaA))/ sigma_square;
}
//' @export
// [[Rcpp::export]]
double neg_loglik_linear_i_cpp(const arma::vec &response_i, const arma::mat &A, const arma::vec &theta_i,
                               double sigma_square){
  arma::vec tmp = A * theta_i;
  //return log(sigma_square) + 1.0 / J / sigma_square * arma::accu(arma::square(response_i - tmp));
  return arma::accu(arma::square(response_i - tmp))/ sigma_square;
}
//' @export
// [[Rcpp::export]]
arma::vec grad_neg_loglik_linear_thetai_cpp(const arma::vec &response_i, const arma::mat &A,
                                            const arma::vec &theta_i, double sigma_square){
  arma::vec tmp = A * theta_i;
  //return - 2.0 / J / sigma_square * A.t() * (response_i - tmp);
  return -2.0 * A.t() * (response_i - tmp) / sigma_square;
}

//' @export
// [[Rcpp::export]]
arma::mat Update_theta_linear_cpp(const arma::mat &theta0, const arma::mat &response, const arma::mat &A0,
                                  double sigma_square, double C){
  arma::mat theta1 = theta0.t();
  int N = response.n_rows;
#pragma omp parallel for
  for(int i=0;i<N;++i){
    double step = 10;
    arma::vec theta0_i_tmp = theta0.row(i).t();
    arma::vec theta1_i_tmp = theta1.col(i);
    arma::vec h = grad_neg_loglik_linear_thetai_cpp(response.row(i).t(), A0, theta0_i_tmp, sigma_square);
    theta1_i_tmp = theta0_i_tmp - step * h;
    theta1_i_tmp = prox_func_cpp(theta1_i_tmp, C);
    while(neg_loglik_linear_i_cpp(response.row(i).t(), A0, theta1_i_tmp, sigma_square) >
            neg_loglik_linear_i_cpp(response.row(i).t(), A0, theta0_i_tmp, sigma_square) &&
          step > 1e-7){
      step *= 0.5;
      theta1_i_tmp = theta0_i_tmp - step * h;
      theta1_i_tmp = prox_func_cpp(theta1_i_tmp, C);
    }
    if(step <= 1e-7){
      Rprintf("warning, threshould of step size of theta needs to be smaller. It's likely the tol setting is too small.\n");
    }
    while(neg_loglik_linear_i_cpp(response.row(i).t(), A0, theta0_i_tmp, sigma_square) -
            neg_loglik_linear_i_cpp(response.row(i).t(), A0, theta1_i_tmp, sigma_square) > 100){
      theta0_i_tmp = theta1_i_tmp;
      h = grad_neg_loglik_linear_thetai_cpp(response.row(i).t(), A0, theta0_i_tmp, sigma_square);
      theta1_i_tmp = theta0_i_tmp - step * h;
      theta1_i_tmp = prox_func_cpp(theta1_i_tmp, C);
      while(neg_loglik_linear_i_cpp(response.row(i).t(), A0, theta1_i_tmp, sigma_square) >
              neg_loglik_linear_i_cpp(response.row(i).t(), A0, theta0_i_tmp, sigma_square) &&
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
//' @export
// [[Rcpp::export]]
double neg_loglik_linear_j_cpp(const arma::vec &response_j, const arma::vec &A_j, const arma::mat &theta,
                               double sigma_square){
  arma::vec tmp = theta * A_j;
  return arma::accu(arma::square(response_j - tmp)) / sigma_square;
}
//' @export
// [[Rcpp::export]]
arma::vec grad_neg_loglik_linear_A_j_cpp(const arma::vec &response_j, const arma::vec &A_j, const arma::vec &Q_j,
                                         const arma::mat &theta, double sigma_square){
  arma::vec tmp = theta * A_j;
  return -(2.0 / sigma_square * theta.t() * (response_j - tmp)) % Q_j;
}

//' @export
// [[Rcpp::export]]
arma::mat Update_A_linear_cpp(const arma::mat &A0, const arma::mat &Q, const arma::mat &response,
                              const arma::mat &theta1, double sigma_square, double C){
  arma::mat A1 = A0.t();
  int J = A0.n_rows;
#pragma omp parallel for
  for(int j=0;j<J;++j){
    double step = 5;
    arma::vec A0_j_tmp = A0.row(j).t();
    arma::vec A1_j_tmp = A1.col(j);
    arma::vec h = grad_neg_loglik_linear_A_j_cpp(response.col(j), A0_j_tmp, Q.row(j).t(), theta1, sigma_square);
    A1_j_tmp = A0_j_tmp - step * h;
    A1_j_tmp = prox_func_cpp(A1_j_tmp, C);
    while(neg_loglik_linear_j_cpp(response.col(j), A1_j_tmp, theta1, sigma_square) >
            neg_loglik_linear_j_cpp(response.col(j), A0_j_tmp, theta1, sigma_square) &&
          step > 1e-7){
      step *= 0.5;
      A1_j_tmp = A0_j_tmp - step * h;
      A1_j_tmp = prox_func_cpp(A1_j_tmp, C);
    }
    if(step <= 1e-7){
      Rprintf("warning, threshould of step size of A needs to be smaller. It's likely the tol setting is too small.\n");
    }
    while(neg_loglik_linear_j_cpp(response.col(j), A0_j_tmp, theta1, sigma_square) -
            neg_loglik_linear_j_cpp(response.col(j), A1_j_tmp, theta1, sigma_square) > 100){
      A0_j_tmp = A1_j_tmp;
      h = grad_neg_loglik_linear_A_j_cpp(response.col(j), A0_j_tmp, Q.row(j).t(), theta1, sigma_square);
      A1_j_tmp = A0_j_tmp - step * h;
      A1_j_tmp = prox_func_cpp(A1_j_tmp, C);
      while(neg_loglik_linear_j_cpp(response.col(j), A1_j_tmp, theta1, sigma_square) >
              neg_loglik_linear_j_cpp(response.col(j), A0_j_tmp, theta1, sigma_square) &&
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
// double Update_sigma_square_cpp(const arma::mat &response, const arma::mat &A1, const arma::mat &Q,
//                                const arma::mat &theta1, double sigma_square){
//   int N = response.n_rows;
//   int J = response.n_cols;
//   arma::mat thetaA = theta1 * A1.t();
//   double step = 10;
//   double h = 1.0 / sigma_square -
//     1.0 / N / J / sigma_square / sigma_square * arma::accu(arma::square(response - thetaA));
//   double sigma_square1 = sigma_square - step * h;
//   while(sigma_square1 < 0 || ((neg_loglik_linear(thetaA, response, sigma_square1) >
//           neg_loglik_linear(thetaA, response, sigma_square)) && step > 1e-3)){
//     step *= 0.5;
//     sigma_square1 = sigma_square - step * h;
//     if(step <= 1e-3){
//       Rprintf("error in update sigma_square\n");
//     }
//   }
//   while(neg_loglik_linear(thetaA, response, sigma_square) -
//            neg_loglik_linear(thetaA, response, sigma_square1) > 1e-3){
//     sigma_square = sigma_square1;
//     h = 1.0 / sigma_square -
//       1.0 / N / J / sigma_square / sigma_square * arma::accu(arma::square(response - thetaA));
//     sigma_square1 = sigma_square - step * h;
//     while(sigma_square1 < 0 || ((neg_loglik_linear(thetaA, response, sigma_square1) >
//                                    neg_loglik_linear(thetaA, response, sigma_square)) && step > 1e-3)){
//       step *= 0.5;
//       sigma_square1 = sigma_square - step * h;
//       if(step <= 1e-3){
//         Rprintf("error in update sigma_square\n");
//       }
//     }
//   }
//   return sigma_square1;
// }

//' @export
// [[Rcpp::export]]
Rcpp::List CJMLE_linear( const arma::mat &response, arma::mat theta0, arma::mat A0,
                                     const arma::mat &Q, double sigma_square, double C, double tol,
                                     bool parallel){
  arma::mat theta1 = Update_theta_linear_cpp(theta0, response, A0, sigma_square, C);
  arma::mat A1 = Update_A_linear_cpp(A0, Q, response, theta1, sigma_square, C);
  double eps = neg_loglik_linear(theta0*A0.t(), response, sigma_square) -
    neg_loglik_linear(theta1*A1.t(), response, sigma_square);
  Rprintf("eps: %f\n", eps);
  while(eps > tol){
    Rcpp::checkUserInterrupt();
    theta0 = theta1;
    A0 = A1;
    theta1 = Update_theta_linear_cpp(theta0, response, A0, sigma_square, C);
    A1 = Update_A_linear_cpp(A0, Q, response, theta1, sigma_square, C);
    eps = neg_loglik_linear(theta0*A0.t(), response, sigma_square) -
      neg_loglik_linear(theta1*A1.t(), response, sigma_square);
    Rprintf("eps: %f\n", eps);
  }
  return Rcpp::List::create(Rcpp::Named("A") = A1,
                            Rcpp::Named("theta") = theta1,
                            Rcpp::Named("sigma_square") = sigma_square,
                            Rcpp::Named("obj") = neg_loglik_linear(theta1*A1.t(), response, sigma_square));
}
