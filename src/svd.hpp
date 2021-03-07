#ifndef TENSORFACT_SVD_HPP
#define TENSORFACT_SVD_HPP

#include <armadillo>

namespace tensorfact {

/// Truncated SVD given error tolerance
template <typename Real>
void TruncatedSvd(const arma::Mat<Real> &A, Real delta, arma::Mat<Real> &U,
                  arma::Col<Real> &s, arma::Mat<Real> &V, arma::uword &rank);

}  // namespace tensorfact

#endif
