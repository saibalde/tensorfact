#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

template <class Real>
void ThinRq(long m, long n, std::vector<Real> &A, std::vector<Real> &R,
            std::vector<Real> &Q);

template <class Real>
void TruncatedSvd(long m, long n, std::vector<Real> &A, Real tolerance,
                  bool is_relative, long &r, std::vector<Real> &U,
                  std::vector<Real> &s, std::vector<Real> &Vt);

#endif
