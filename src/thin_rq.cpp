#include "thin_rq.hpp"

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

template void ThinRq<float>(long m, long n, std::vector<float> &A,
                            std::vector<float> &R, std::vector<float> &Q);
template void ThinRq<double>(long m, long n, std::vector<double> &A,
                             std::vector<double> &R, std::vector<double> &Q);
