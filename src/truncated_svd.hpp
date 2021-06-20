#ifndef TRUNCATEDSVD_HPP
#define TRUNCATEDSVD_HPP

#include <vector>

template <class Real>
void TruncatedSvd(long m, long n, std::vector<Real> &A, Real tolerance,
                  bool is_relative, long &r, std::vector<Real> &U,
                  std::vector<Real> &s, std::vector<Real> &Vt);

#endif
