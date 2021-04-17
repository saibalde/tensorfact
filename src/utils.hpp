#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

void ThinRq(int m, int n, std::vector<double> &A, std::vector<double> &R,
            std::vector<double> &Q);

void TruncatedSvd(int m, int n, std::vector<double> &A, double tolerance,
                  bool is_relative, int &r, std::vector<double> &U,
                  std::vector<double> &s, std::vector<double> &Vt);

#endif
