#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

void ThinRq(long m, long n, std::vector<double> &A, std::vector<double> &R,
            std::vector<double> &Q);

void TruncatedSvd(long m, long n, std::vector<double> &A, double tolerance,
                  bool is_relative, long &r, std::vector<double> &U,
                  std::vector<double> &s, std::vector<double> &Vt);

#endif
