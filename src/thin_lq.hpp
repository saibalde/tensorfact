#ifndef THINRQ_HPP
#define THINRQ_HPP

#include <armadillo>

template <class Real>
void ThinLq(const arma::Mat<Real> &A, arma::Mat<Real> &L, arma::Mat<Real> &Q);

#endif
