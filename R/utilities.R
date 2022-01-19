#' @export
standardization_cjmle <- function(theta0, A0, d0){
  theta0 <- as.matrix(theta0)
  A0 <- as.matrix(A0)
  N <- nrow(theta0)
  d1 <- d0 + 1/N * A0 %*% t(theta0) %*% matrix(rep(1,N),N)
  theta1 <- theta0 - 1/N * matrix(rep(1,N),N) %*% matrix(rep(1,N),1) %*% theta0
  A1 <- A0
  return(list("theta1"=theta1, "A1"=A1, "d1"=d1))
}