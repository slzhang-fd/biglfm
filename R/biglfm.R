#' Joint maximum likelihood estimator for general latent factor model fitting to large scale dataset.
#' @param model one of the three models: "linear", "mirt", "poisson" must be selected.
#' @param response N by J 0-1 matrix, each row represents test takers' response to J items.
#' @param theta0 N by K matrix, initial values of latent traits parameters.
#' @param A0 J by K matrix, initial values of loading matrix paramters.
#' @param Q J by K matrix, pre-specified Q matrix for structured information, if Q is set to NULL, the algorithm performs 
#' exploretory setting.
#' @param C2 Constraint constant, for more information see to the paper \url{www.google.com}.
#' @param tol Tolerance threshold, default value is 1e-4.
#' @param parallel If enable parallel computing, default value is True, i.e. use all the computing cores in your computer.
#' @return A list contains the estimated loading matrix A and latent traits matrix theta.
#' @examples 
#' \dontrun{
#' library(biglfm)
#' # To fit the Linear model
#' res_linear <- biglfm(model = "linear", response_linear,
#'  theta, A, Q, 1.44*C2, tol = 1e-4, parallel = T)
#' 
#' # MIRT model
#' res_mirt <- biglfm(model = "mirt",response_mirt, theta, A, Q, 1.44*C2, tol = 1e-4, parallel =T)
#' 
#' # Poisson modeel
#' res_poi <- biglfm(model = "poisson",response_poi, theta, A, Q, 1.44*C2, tol = 1e-4, parallel = T)
#' }
#' @useDynLib biglfm, .registration = TRUE
#' @importFrom Rcpp evalCpp sourceCpp
#' @export
biglfm <- function(model, response, theta0, A0, Q, C2, tol=1e-4, parallel=T){
  if(is.null(Q))
    Q <- matrix(1, nrow(A0), ncol(A0))
  if(model == "linear")
    result <- CJMLE_linear(response = response, theta0=theta0,
                           A0=A0, Q=Q, sigma_square0=1, C=C2, tol = tol, parallel=parallel)
  else if(model == "mirt")
    result <- CJMLE_MIRT(response=response, nonmis_ind = 1-is.na(response), theta0 = theta0,
                         A0=A0, Q=Q, C=C2, tol = tol, parallel=parallel)
  else if(model == "poisson")
    result <- CJMLE_poisson(response=response, theta0 = theta0, A0=A0, Q=Q, C=C2, tol = tol, parallel=parallel)
  else if(model == "hellomp")
    hellomp(parallel)
  else
    stop("model should be in one of 'linear', 'mirt' and 'poisson'.")
  return(result)
}

.onUnload <- function (libpath) {
  library.dynam.unload("biglfm", libpath)
}
.onAttach <- function(library, pkg)
{
  Rv <- R.Version()
  if(!exists("getRversion", baseenv()) || (getRversion() < "3.1.2"))
    stop("This package requires R 3.1.2 or later")
  assign(".biglfm.home", file.path(library, pkg),
         pos=match("package:biglfm", search()))
  ars.version <- "1.1.3 (2019-03-05)"
  assign(".biglfm.version", ars.version, pos=match("package:biglfm", search()))
  if(interactive())
  {
    packageStartupMessage(paste("Package 'biglfm', ", ars.version, ". ",sep=""),appendLF=TRUE)
    packageStartupMessage("Type 'help(biglfm-package)' for package information",appendLF=TRUE)
  }
  invisible()
}
