#include "utils.hpp"

#include <blas.hh>
#include <cmath>
#include <lapack.hh>
#include <stdexcept>

template <class Real>
void ThinRq(long m, long n, std::vector<Real> &A, std::vector<Real> &R,
            std::vector<Real> &Q) {
    if (m < 1 || n < 1) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }

    if (A.size() != m * n) {
        throw std::invalid_argument(
            "Matrix dimensions are incompatible with number of entries");
    }

    const long k = std::min(m, n);

    std::vector<Real> tau(k);

    const long status1 = lapack::gerqf(m, n, A.data(), m, tau.data());
    if (status1 != 0) {
        throw std::runtime_error("RQ factorization was not successful");
    }

    R.resize(m * k);
    for (long j = 0; j < k; ++j) {
        for (long i = 0; i < m; ++i) {
            if (i <= j + m - k) {
                if (m < n) {
                    R[i + j * m] = A[i + (n - m + j) * m];
                } else {
                    R[i + j * m] = A[i + j * m];
                }
            } else {
                R[i + j * m] = static_cast<Real>(0);
            }
        }
    }

    Q.resize(k * n);
    for (long j = 0; j < n; ++j) {
        for (long i = 0; i < k; ++i) {
            if (m < n) {
                Q[i + j * k] = A[i + j * m];
            } else {
                Q[i + j * k] = A[m - k + i + j * m];
            }
        }
    }

    const long status2 = lapack::ungrq(k, n, k, Q.data(), k, tau.data());
    if (status2 != 0) {
        throw std::runtime_error(
            "Q computation in RQ factorization was not successful");
    }
}

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

    std::vector<Real> U_thin(m * k);
    std::vector<Real> s_thin(k);
    std::vector<Real> Vt_thin(k * n);

    {
        const long status =
            lapack::gesdd(lapack::Job::SomeVec, m, n, A.data(), m,
                          s_thin.data(), U_thin.data(), m, Vt_thin.data(), k);

        if (status != 0) {
            throw std::runtime_error("SVD computation failed");
        }
    }

    Real frobenius_max_error = static_cast<Real>(0);
    if (is_relative) {
        for (long i = 0; i < k; ++i) {
            frobenius_max_error += std::pow(s_thin[i], 2);
        }

        frobenius_max_error *= std::pow(tolerance, 2);
    } else {
        frobenius_max_error = std::pow(tolerance, 2);
    }

    Real frobenius_error = static_cast<Real>(0);
    r = k;
    while (r > 0) {
        frobenius_error += std::pow(s_thin[r - 1], 2);
        if (frobenius_error > frobenius_max_error) {
            break;
        }
        --r;
    }

    U.resize(m * r);
    for (long j = 0; j < r; ++j) {
        for (long i = 0; i < m; ++i) {
            U[i + j * m] = U_thin[i + j * m];
        }
    }

    s.resize(r);
    for (long i = 0; i < r; ++i) {
        s[i] = s_thin[i];
    }

    Vt.resize(r * n);
    for (long j = 0; j < n; ++j) {
        for (long i = 0; i < r; ++i) {
            Vt[i + j * r] = Vt_thin[i + j * k];
        }
    }
}

template void ThinRq<float>(long m, long n, std::vector<float> &A,
                            std::vector<float> &R, std::vector<float> &Q);
template void ThinRq<double>(long m, long n, std::vector<double> &A,
                             std::vector<double> &R, std::vector<double> &Q);

template void TruncatedSvd<float>(long m, long n, std::vector<float> &A,
                                  float tolerance, bool is_relative, long &r,
                                  std::vector<float> &U, std::vector<float> &s,
                                  std::vector<float> &Vt);
template void TruncatedSvd<double>(long m, long n, std::vector<double> &A,
                                   double tolerance, bool is_relative, long &r,
                                   std::vector<double> &U,
                                   std::vector<double> &s,
                                   std::vector<double> &Vt);
