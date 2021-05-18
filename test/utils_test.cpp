#include "utils.hpp"

#include <gtest/gtest.h>

#include <blas.hh>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

template <typename Real>
void CreateRandomMatrix(long m, long n, std::vector<Real> &A) {
    if (m < 1 || n < 1) {
        throw std::invalid_argument(
            "Number of rows and columns must be positive");
    }

    A.resize(m * n);

    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<Real> distribution(-1, 1);

    for (long i = 0; i < m * n; ++i) {
        A[i] = distribution(generator);
    }
}

template <typename Real>
void ThinRqTest(long m, long n) {
    std::vector<Real> A;
    CreateRandomMatrix<Real>(m, n, A);

    std::vector<Real> C = A;

    std::vector<Real> R;
    std::vector<Real> Q;
    ThinRq<Real>(m, n, A, R, Q);

    const long k = std::min(m, n);

    ASSERT_EQ(R.size(), m * k);
    ASSERT_EQ(Q.size(), k * n);

    for (long j = 0; j < k; ++j) {
        for (long i = j + m - k + 1; i < m; ++i) {
            ASSERT_NEAR(R[i + j * m], static_cast<Real>(0),
                        10 * std::numeric_limits<Real>::epsilon());
        }
    }

    for (long i1 = 0; i1 < k; ++i1) {
        for (long i2 = 0; i2 < i1; ++i2) {
            Real dot_product = 0.0;
            for (long j = 0; j < n; ++j) {
                dot_product += Q[i1 + j * k] * Q[i2 + j * k];
            }

            ASSERT_NEAR(dot_product, static_cast<Real>(0),
                        10 * std::numeric_limits<Real>::epsilon());
        }

        Real norm_squared = static_cast<Real>(0);
        for (long j = 0; j < n; ++j) {
            norm_squared += std::pow(Q[i1 + j * k], 2);
        }

        ASSERT_NEAR(norm_squared, static_cast<Real>(1),
                    10 * std::numeric_limits<Real>::epsilon());
    }

    std::vector<Real> B(m * n);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m,
               n, k, static_cast<Real>(1), R.data(), m, Q.data(), k,
               static_cast<Real>(0), B.data(), m);

    Real frobenius_error = static_cast<Real>(0);
    for (long j = 0; j < n; ++j) {
        for (long i = 0; i < m; ++i) {
            frobenius_error += std::pow(B[i + j * m] - C[i + j * m], 2);
        }
    }
    frobenius_error = std::sqrt(frobenius_error);
    ASSERT_NEAR(frobenius_error, static_cast<Real>(0),
                100 * std::numeric_limits<Real>::epsilon());
}

TEST(utils, ThinRq) {
    ThinRqTest<float>(4, 64);
    ThinRqTest<double>(4, 64);

    ThinRqTest<float>(16, 16);
    ThinRqTest<double>(16, 16);

    ThinRqTest<float>(64, 4);
    ThinRqTest<double>(64, 4);
}

template <typename Real>
void SvdFactorQualityTest(long m, long n, long r, const std::vector<Real> &U,
                          const std::vector<Real> &s,
                          const std::vector<Real> &Vt) {
    ASSERT_EQ(U.size(), m * r);
    ASSERT_EQ(s.size(), r);
    ASSERT_EQ(Vt.size(), r * n);

    for (long j1 = 0; j1 < r; ++j1) {
        for (long j2 = 0; j2 < j1; ++j2) {
            Real dot_product = static_cast<Real>(0);
            for (long i = 0; i < m; ++i) {
                dot_product += U[i + j1 * m] * U[i + j2 * m];
            }

            ASSERT_NEAR(dot_product, static_cast<Real>(0),
                        100 * std::numeric_limits<Real>::epsilon());
        }

        Real norm_squared = static_cast<Real>(0);
        for (long i = 0; i < m; ++i) {
            norm_squared += std::pow(U[i + j1 * m], 2);
        }

        ASSERT_NEAR(norm_squared, static_cast<Real>(1),
                    100 * std::numeric_limits<Real>::epsilon());
    }

    for (long i = 0; i < r; ++i) {
        if (i < r - 1) {
            ASSERT_GE(s[i], s[i + 1]);
        } else {
            ASSERT_GT(s[i], static_cast<Real>(0));
        }
    }

    for (long i1 = 0; i1 < r; ++i1) {
        for (long i2 = 0; i2 < i1; ++i2) {
            Real dot_product = static_cast<Real>(0);
            for (long j = 0; j < n; ++j) {
                dot_product += Vt[i1 + j * r] * Vt[i2 + j * r];
            }

            ASSERT_NEAR(dot_product, static_cast<Real>(0),
                        100 * std::numeric_limits<Real>::epsilon());
        }

        Real norm_squared = static_cast<Real>(0);
        for (long j = 0; j < n; ++j) {
            norm_squared += std::pow(Vt[i1 + j * r], 2);
        }

        ASSERT_NEAR(norm_squared, static_cast<Real>(1),
                    100 * std::numeric_limits<Real>::epsilon());
    }
}

template <typename Real>
void TruncatedSvdTest(long m, long n, long r) {
    if (m < 1 || n < 1 || r < 1) {
        throw std::invalid_argument("Matrix sizes and rank must be positive");
    }

    if (r > m || r > n) {
        throw std::invalid_argument("Rank must be smaller than matrix size");
    }

    // Construct appropriate matrix
    std::vector<Real> A(m * n);
    long k;
    Real absolute_tolerance;
    Real relative_tolerance;

    {
        // Create Ut factor
        std::vector<Real> Ut;
        {
            std::vector<Real> temp;
            CreateRandomMatrix<Real>(r, m, temp);

            std::vector<Real> R;
            ThinRq<Real>(r, m, temp, R, Ut);
        }

        // Create s factor; s[i] ~ Uniform([2 * (r - 1 - i), 2 * (r - i)])
        std::vector<Real> s;
        CreateRandomMatrix<Real>(r, 1, s);
        for (long i = 0; i < r; ++i) {
            s[i] += 2 * (r - 1 - i) + 1;
        }

        // Create Vt factor
        std::vector<Real> Vt;
        {
            std::vector<Real> temp;
            CreateRandomMatrix<Real>(r, n, temp);

            std::vector<Real> R;
            ThinRq<Real>(r, n, temp, R, Vt);
        }

        // Compute A = Ut.conjugate() * diagmat(s) * Vt
        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < r; ++i) {
                Vt[i + j * r] *= s[i];
            }
        }

        blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                   m, n, r, static_cast<Real>(1), Ut.data(), r, Vt.data(), r,
                   static_cast<Real>(0), A.data(), m);

        // Compute absolute and relative errors given specified maximum rank
        std::vector<Real> absolute_error(r);

        absolute_error[r - 1] = static_cast<Real>(0);
        for (long i = r - 1; i > 0; --i) {
            absolute_error[i - 1] = absolute_error[i] + std::pow(s[i], 2);
        }
        Real frobenius_norm = absolute_error[0] + std::pow(s[0], 2);

        for (long i = 0; i < r; ++i) {
            absolute_error[i] = std::sqrt(absolute_error[i]);
        }
        frobenius_norm = std::sqrt(frobenius_norm);

        std::vector<Real> relative_error(r);

        for (long i = 0; i < r; ++i) {
            relative_error[i] = absolute_error[i] / frobenius_norm;
        }

        // Determine 75% Frobenius norm cutoff rank
        k = 1;
        while (k <= r) {
            if (relative_error[k - 1] <= static_cast<Real>(0.25)) {
                break;
            }
            ++k;
        }

        // Compute accuracy levels based on cutoff rank
        absolute_tolerance =
            (absolute_error[k - 2] + absolute_error[k - 1]) / 2;
        relative_tolerance =
            (relative_error[k - 2] + relative_error[k - 1]) / 2;
    }

    // Test thin SVD
    {
        std::vector<Real> B = A;

        std::vector<Real> U;
        std::vector<Real> s;
        std::vector<Real> Vt;
        long r;
        TruncatedSvd<Real>(m, n, B, static_cast<Real>(0), false, r, U, s, Vt);

        ASSERT_EQ(r, std::min(m, n));

        SvdFactorQualityTest<Real>(m, n, r, U, s, Vt);

        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < r; ++i) {
                Vt[i + j * r] *= s[i];
            }
        }

        std::vector<Real> C(m * n);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   m, n, r, static_cast<Real>(1), U.data(), m, Vt.data(), r,
                   static_cast<Real>(0), C.data(), m);

        Real frobenius_error = static_cast<Real>(0);
        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < m; ++i) {
                frobenius_error += std::pow(A[i + j * m] - C[i + j * m], 2);
            }
        }
        frobenius_error = std::sqrt(frobenius_error);
        ASSERT_LE(frobenius_error, 1000 * std::numeric_limits<Real>::epsilon());
    }

    // Test truncated SVD with absolute error
    {
        std::vector<Real> B = A;

        std::vector<Real> U;
        std::vector<Real> s;
        std::vector<Real> Vt;
        long r;
        TruncatedSvd<Real>(m, n, B, absolute_tolerance, false, r, U, s, Vt);

        ASSERT_EQ(r, k);

        SvdFactorQualityTest<Real>(m, n, r, U, s, Vt);

        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < k; ++i) {
                Vt[i + j * r] *= s[i];
            }
        }

        std::vector<Real> C(m * n);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   m, n, r, static_cast<Real>(1), U.data(), m, Vt.data(), r,
                   static_cast<Real>(0), C.data(), m);

        Real frobenius_error = static_cast<Real>(0);
        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < m; ++i) {
                frobenius_error += std::pow(A[i + j * m] - C[i + j * m], 2);
            }
        }
        frobenius_error = std::sqrt(frobenius_error);
        ASSERT_LE(frobenius_error, absolute_tolerance);
    }

    // Test thin SVD
    {
        std::vector<Real> B = A;

        std::vector<Real> U;
        std::vector<Real> s;
        std::vector<Real> Vt;
        long r;
        TruncatedSvd<Real>(m, n, B, relative_tolerance, true, r, U, s, Vt);

        ASSERT_EQ(r, k);

        SvdFactorQualityTest<Real>(m, n, r, U, s, Vt);

        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < k; ++i) {
                Vt[i + j * r] *= s[i];
            }
        }

        std::vector<Real> C(m * n);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   m, n, r, static_cast<Real>(1), U.data(), m, Vt.data(), r,
                   static_cast<Real>(0), C.data(), m);

        Real frobenius_error = static_cast<Real>(0);
        Real frobenius_norm = static_cast<Real>(0);
        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < m; ++i) {
                frobenius_error += std::pow(A[i + j * m] - C[i + j * m], 2);
                frobenius_norm += std::pow(A[i + j * m], 2);
            }
        }
        frobenius_error = std::sqrt(frobenius_error);
        frobenius_norm = std::sqrt(frobenius_norm);
        ASSERT_LE(frobenius_error, relative_tolerance * frobenius_norm);
    }
}

TEST(utils, TruncatedSvd) {
    TruncatedSvdTest<float>(16, 64, 8);
    TruncatedSvdTest<double>(16, 64, 8);

    TruncatedSvdTest<float>(32, 32, 8);
    TruncatedSvdTest<double>(32, 32, 8);

    TruncatedSvdTest<float>(64, 16, 8);
    TruncatedSvdTest<double>(64, 16, 8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
