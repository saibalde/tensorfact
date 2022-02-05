#ifndef TRUNCATEDSVD_HPP
#define TRUNCATEDSVD_HPP

#include <armadillo>
#include <vector>

template <class Real>
void TruncatedSvd(long m, long n, std::vector<Real> &A, Real tolerance,
                  bool is_relative, long &r, std::vector<Real> &U,
                  std::vector<Real> &s, std::vector<Real> &Vt);

template <class Real>
void TruncatedSvd(const arma::Mat<Real> &A, Real tolerance, bool is_relative,
                  arma::Mat<Real> &U, arma::Col<Real> &s, arma::Mat<Real> &V,
                  long &r);

#endif
