#' Responses of 2500 people for 100 items with 5 latent traits.
#'
#' A toy example dataset generated from latent factor model.
#'
#' @format A list with 7 elements:
#' \describe{
#'    \item{A}{loading matrix, 100 by 5.}
#'    \item{theta}{latent traits matrix, 2500 by 5.}
#'    \item{Q}{Q matrix, 100 by 5 with elements 0 or 1.}
#'    \item{C2}{squared constraint constant, double.}
#'    \item{response_linear}{matrix, continuous responses of 2500 people for 100 items generated from linear factor model.}
#'    \item{response_mirt}{matrix, binary responses of 2500 people for 100 items generated from mirt factor model.}
#'    \item{response_poi}{matrix, positive integer responses of 2500 people for 100 items generated from Poisson factor model.}}
"test_data"