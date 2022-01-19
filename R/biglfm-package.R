#' Joint maximum likelihood estimator for general latent factor model fitting to large scale dataset
#' 
#' 
#' Joint maximum likelihood estimation for general latent factor model for
#' ultrahigh-dimensional data sets that cannot be analysised by the ordianry marginalized 
#' likelihood method.
#' This package utilizes alternating minimization algorithm and OpenMP parallel computing (if supported)
#' during model fitting. As a result, this
#' package is computation-efficient and highly scalable
#' as compared to existing latent factor model packages such as
#' \href{https://CRAN.R-project.org/package=mirt}{mirt} and
#' \href{https://CRAN.R-project.org/package=DCM}{DCM}, thus allowing for
#' powerful big data analysis even with only an ordinary laptop.
#' 
#' 
#' \tabular{ll}{ Package: \tab biglfm\cr Type: \tab Package\cr Version: \tab
#' 1.1.1\cr Date: \tab 2017-12-16\cr License: \tab GPL-3\cr}
#' 
#' Penalized regression models, in particular the lasso, have been extensively
#' applied to analyzing high-dimensional data sets. However, due to the memory
#' limit, existing R packages are not capable of fitting lasso models for
#' ultrahigh-dimensional, multi-gigabyte data sets which have been increasingly
#' seen in many areas such as genetics, biomedical imaging, genome sequencing
#' and high-frequency finance.
#' 
#' This package aims to fill the gap by extending lasso model fitting to Big
#' Data in R. Version >= 1.2-3 represents a major redesign where the source
#' code is converted into C++ (previously in C), and new feature screening
#' rules, as well as OpenMP parallel computing, are implemented. Some key
#' features of \code{biglasso} are summarized as below: \enumerate{ \item it
#' utilizes memory-mapped files to store the massive data on the disk, only
#' loading data into memory when necessary during model fitting. Consequently,
#' it's able to seamlessly data-larger-than-RAM cases. \item it is built upon
#' pathwise coordinate descent algorithm with warm start, active set cycling,
#' and feature screening strategies, which has been proven to be one of fastest
#' lasso solvers. \item in incorporates our newly developed hybrid hybrid
#' safe-strong rules that outperform state-of-the-art screening rules such as
#' the sequential strong rule (SSR) and the sequential EDPP rule (SEDPP) with
#' additional 1.5x to 4x speedup. \item the implementation is designed to be as
#' memory-efficient as possible by eliminating extra copies of the data created
#' by other R packages, making it at least 2x more memory-efficient than
#' \code{glmnet}. \item the underlying computation is implemented in C++, and
#' parallel computing with OpenMP is also supported. }
#' 
#' @name biglfm-package
#' @docType package
#' @author Siliang Zhang, Yunxiao Chen and Xiaoou Li
#' 
#' Maintainer: Siliang Zhang <zhangsiliang123@gmail.com>
#' @references \itemize{
#' \item Chen, Y., Li, X., and Zhang, S. (2017). Joint maximum likelihood 
#' estimation for high-dimensional exploratory item response analysis. Unpublished Manuscript.
#' \item Chen, Y., Li, X., and Zhang, S. (2017). Structured Latent Factor Analysis 
#' for Large-scale Data: Identifiability and Its Implications. }
#' @keywords package
NULL