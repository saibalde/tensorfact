#include "truncated_svd.hpp"

#include <armadillo>
#include <cmath>
#include <stdexcept>

template <class Real>
void TruncatedSvd(long m, long n, std::vector<Real> &A, Real tolerance,
                  bool is_relative, long &r, std::vector<Real> &U,
                  std::vector<Real> &s, std::vector<Real> &Vt) {
    if (m < 1 || n < 1) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }

    if (A.size() != m * n) {
        throw std::invalid_argument(
            "Matrix dimensions are incompatible with number of entries");
    }

    const long k = std::min(m, n);

    arma::Mat<Real> A_temp(m, n);
    for (long j = 0; j < n; ++j) {
        for (long i = 0; i < m; ++i) {
            A_temp(i, j) = A[i + j * m];
        }
    }

    arma::Mat<Real> U_thin;
    arma::Col<Real> s_thin;
    arma::Mat<Real> V_thin;

    bool status = arma::svd_econ(U_thin, s_thin, V_thin, A_temp);

    if (!status) {
        throw std::runtime_error("SVD decomposition failed");
    }

    Real frobenius_max_error = static_cast<Real>(0);
    if (is_relative) {
        for (long i = 0; i < k; ++i) {
            frobenius_max_error += std::pow(s_thin(i), 2);
        }

        frobenius_max_error *= std::pow(tolerance, 2);
    } else {
        frobenius_max_error = std::pow(tolerance, 2);
    }

    Real frobenius_error = static_cast<Real>(0);
    r = k;
    while (r > 0) {
        frobenius_error += std::pow(s_thin(r - 1), 2);
        if (frobenius_error > frobenius_max_error) {
            break;
        }
        --r;
    }

    U.resize(m * r);
    for (long j = 0; j < r; ++j) {
        for (long i = 0; i < m; ++i) {
            U[i + j * m] = U_thin(i, j);
        }
    }

    s.resize(r);
    for (long i = 0; i < r; ++i) {
        s[i] = s_thin(i);
    }

    Vt.resize(r * n);
    for (long j = 0; j < n; ++j) {
        for (long i = 0; i < r; ++i) {
            Vt[i + j * r] = V_thin(j, i);
        }
    }
}

template <class Real>
void TruncatedSvd(const arma::Mat<Real> &A, Real tolerance, bool is_relative,
                  arma::Mat<Real> &U, arma::Col<Real> &s, arma::Mat<Real> &V,
                  long &r) {
    arma::Mat<Real> U_thin;
    arma::Col<Real> s_thin;
    arma::Mat<Real> V_thin;
    bool status = arma::svd_econ(U_thin, s_thin, V_thin, A);
    if (!status) {
        throw std::runtime_error("SVD decomposition failed");
    }

    Real frobenius_max_error = static_cast<Real>(0);
    if (is_relative) {
        for (long i = 0; i < s_thin.n_rows; ++i) {
            frobenius_max_error += std::pow(s_thin(i), 2);
        }

        frobenius_max_error *= std::pow(tolerance, 2);
    } else {
        frobenius_max_error = std::pow(tolerance, 2);
    }

    Real frobenius_error = static_cast<Real>(0);
    r = s_thin.n_rows;
    while (r > 0) {
        frobenius_error += std::pow(s_thin(r - 1), 2);
        if (frobenius_error > frobenius_max_error) {
            break;
        }
        --r;
    }

    U.set_size(A.n_rows, r);
    for (long j = 0; j < r; ++j) {
        for (long i = 0; i < A.n_rows; ++i) {
            U(i, j) = U_thin(i, j);
        }
    }

    s.set_size(r);
    for (long i = 0; i < r; ++i) {
        s(i) = s_thin(i);
    }

    V.set_size(A.n_cols, r);
    for (long j = 0; j < r; ++j) {
        for (long i = 0; i < A.n_cols; ++i) {
            V(i, j) = V_thin(i, j);
        }
    }
}

template void TruncatedSvd<float>(long m, long n, std::vector<float> &A,
                                  float tolerance, bool is_relative, long &r,
                                  std::vector<float> &U, std::vector<float> &s,
                                  std::vector<float> &Vt);
template void TruncatedSvd<double>(long m, long n, std::vector<double> &A,
                                   double tolerance, bool is_relative, long &r,
                                   std::vector<double> &U,
                                   std::vector<double> &s,
                                   std::vector<double> &Vt);

template void TruncatedSvd<float>(const arma::Mat<float> &A, float tolerance,
                                  bool is_relative, arma::Mat<float> &U,
                                  arma::Col<float> &s, arma::Mat<float> &V,
                                  long &r);

template void TruncatedSvd<double>(const arma::Mat<double> &A, double tolerance,
                                   bool is_relative, arma::Mat<double> &U,
                                   arma::Col<double> &s, arma::Mat<double> &V,
                                   long &r);
