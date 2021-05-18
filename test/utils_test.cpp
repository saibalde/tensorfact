#include "utils.hpp"

#include <gtest/gtest.h>

#include <blas.hh>
#include <cmath>
#include <random>
#include <stdexcept>

void CreateRandomMatrix(long m, long n, std::vector<double> &A) {
    if (m < 1 || n < 1) {
        throw std::invalid_argument(
            "Number of rows and columns must be positive");
    }

    A.resize(m * n);

    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (long i = 0; i < m * n; ++i) {
        A[i] = distribution(generator);
    }
}

void ReducedRqTest(long m, long n) {
    std::vector<double> A;
    CreateRandomMatrix(m, n, A);

    std::vector<double> C = A;

    std::vector<double> R;
    std::vector<double> Q;
    ThinRq(m, n, A, R, Q);

    const long k = std::min(m, n);

    ASSERT_EQ(R.size(), m * k);
    ASSERT_EQ(Q.size(), k * n);

    for (long j = 0; j < k; ++j) {
        for (long i = j + m - k + 1; i < m; ++i) {
            ASSERT_NEAR(R[i + j * m], 0.0, 1.0e-15);
        }
    }

    for (long i1 = 0; i1 < k; ++i1) {
        for (long i2 = 0; i2 < i1; ++i2) {
            double dot_product = 0.0;
            for (long j = 0; j < n; ++j) {
                dot_product += Q[i1 + j * k] * Q[i2 + j * k];
            }

            ASSERT_NEAR(dot_product, 0.0, 1.0e-15);
        }

        double norm_squared = 0.0;
        for (long j = 0; j < n; ++j) {
            norm_squared += std::pow(Q[i1 + j * k], 2.0);
        }

        ASSERT_NEAR(norm_squared, 1.0, 1.0e-15);
    }

    std::vector<double> B(m * n);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m,
               n, k, 1.0, R.data(), m, Q.data(), k, 0.0, B.data(), m);

    double frobenius_error = 0.0;
    for (long j = 0; j < n; ++j) {
        for (long i = 0; i < m; ++i) {
            frobenius_error += std::pow(B[i + j * m] - C[i + j * m], 2.0);
        }
    }
    frobenius_error = std::sqrt(frobenius_error);
    ASSERT_NEAR(frobenius_error, 0.0, 1.0e-14);
}

TEST(utils, ThinRq) {
    ReducedRqTest(4, 64);
    ReducedRqTest(16, 16);
    ReducedRqTest(64, 4);
}

void SvdFactorQualityTest(long m, long n, long r, const std::vector<double> &U,
                          const std::vector<double> &s,
                          const std::vector<double> &Vt) {
    ASSERT_EQ(U.size(), m * r);
    ASSERT_EQ(s.size(), r);
    ASSERT_EQ(Vt.size(), r * n);

    for (long j1 = 0; j1 < r; ++j1) {
        for (long j2 = 0; j2 < j1; ++j2) {
            double dot_product = 0.0;
            for (long i = 0; i < m; ++i) {
                dot_product += U[i + j1 * m] * U[i + j2 * m];
            }

            ASSERT_NEAR(dot_product, 0.0, 1.0e-14);
        }

        double norm_squared = 0.0;
        for (long i = 0; i < m; ++i) {
            norm_squared += std::pow(U[i + j1 * m], 2.0);
        }

        ASSERT_NEAR(norm_squared, 1.0, 1.0e-14);
    }

    for (long i = 0; i < r; ++i) {
        if (i < r - 1) {
            ASSERT_GE(s[i], s[i + 1]);
        } else {
            ASSERT_GT(s[i], 0.0);
        }
    }

    for (long i1 = 0; i1 < r; ++i1) {
        for (long i2 = 0; i2 < i1; ++i2) {
            double dot_product = 0.0;
            for (long j = 0; j < n; ++j) {
                dot_product += Vt[i1 + j * r] * Vt[i2 + j * r];
            }

            ASSERT_NEAR(dot_product, 0.0, 1.0e-14);
        }

        double norm_squared = 0.0;
        for (long j = 0; j < n; ++j) {
            norm_squared += std::pow(Vt[i1 + j * r], 2.0);
        }

        ASSERT_NEAR(norm_squared, 1.0, 1.0e-14);
    }
}

void TruncatedSvdTest(long m, long n, long r) {
    if (m < 1 || n < 1 || r < 1) {
        throw std::invalid_argument("Matrix sizes and rank must be positive");
    }

    if (r > m || r > n) {
        throw std::invalid_argument("Rank must be smaller than matrix size");
    }

    // Construct appropriate matrix
    std::vector<double> A(m * n);
    long k;
    double absolute_tolerance;
    double relative_tolerance;

    {
        // Create Ut factor
        std::vector<double> Ut;
        {
            std::vector<double> temp;
            CreateRandomMatrix(r, m, temp);

            std::vector<double> R;
            ThinRq(r, m, temp, R, Ut);
        }

        // Create s factor; s[i] ~ Uniform([2 * (r - 1 - i), 2 * (r - i)])
        std::vector<double> s;
        CreateRandomMatrix(r, 1, s);
        for (long i = 0; i < r; ++i) {
            s[i] += 2 * (r - 1 - i) + 1;
        }

        // Create Vt factor
        std::vector<double> Vt;
        {
            std::vector<double> temp;
            CreateRandomMatrix(r, n, temp);

            std::vector<double> R;
            ThinRq(r, n, temp, R, Vt);
        }

        // Compute A = Ut.conjugate() * diagmat(s) * Vt
        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < r; ++i) {
                Vt[i + j * r] *= s[i];
            }
        }

        blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                   m, n, r, 1.0, Ut.data(), r, Vt.data(), r, 0.0, A.data(), m);

        // Compute absolute and relative errors given specified maximum rank
        std::vector<double> absolute_error(r);

        absolute_error[r - 1] = 0.0;
        for (long i = r - 1; i > 0; --i) {
            absolute_error[i - 1] = absolute_error[i] + std::pow(s[i], 2.0);
        }
        double frobenius_norm = absolute_error[0] + std::pow(s[0], 2.0);

        for (long i = 0; i < r; ++i) {
            absolute_error[i] = std::sqrt(absolute_error[i]);
        }
        frobenius_norm = std::sqrt(frobenius_norm);

        std::vector<double> relative_error(r);

        for (long i = 0; i < r; ++i) {
            relative_error[i] = absolute_error[i] / frobenius_norm;
        }

        // Determine 75% Frobenius norm cutoff rank
        k = 1;
        while (k <= r) {
            if (relative_error[k - 1] <= 0.25) {
                break;
            }
            ++k;
        }

        // Compute accuracy levels based on cutoff rank
        absolute_tolerance =
            (absolute_error[k - 2] + absolute_error[k - 1]) / 2.0;
        relative_tolerance =
            (relative_error[k - 2] + relative_error[k - 1]) / 2.0;
    }

    // Test thin SVD
    {
        std::vector<double> B = A;

        std::vector<double> U;
        std::vector<double> s;
        std::vector<double> Vt;
        long r;
        TruncatedSvd(m, n, B, 0.0, false, r, U, s, Vt);

        ASSERT_EQ(r, std::min(m, n));

        SvdFactorQualityTest(m, n, r, U, s, Vt);

        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < r; ++i) {
                Vt[i + j * r] *= s[i];
            }
        }

        std::vector<double> C(m * n);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   m, n, r, 1.0, U.data(), m, Vt.data(), r, 0.0, C.data(), m);

        double frobenius_error = 0.0;
        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < m; ++i) {
                frobenius_error += std::pow(A[i + j * m] - C[i + j * m], 2.0);
            }
        }
        frobenius_error = std::sqrt(frobenius_error);
        ASSERT_LE(frobenius_error, 1.0e-13);
    }

    // Test truncated SVD with absolute error
    {
        std::vector<double> B = A;

        std::vector<double> U;
        std::vector<double> s;
        std::vector<double> Vt;
        long r;
        TruncatedSvd(m, n, B, absolute_tolerance, false, r, U, s, Vt);

        ASSERT_EQ(r, k);

        SvdFactorQualityTest(m, n, r, U, s, Vt);

        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < k; ++i) {
                Vt[i + j * r] *= s[i];
            }
        }

        std::vector<double> C(m * n);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   m, n, r, 1.0, U.data(), m, Vt.data(), r, 0.0, C.data(), m);

        double frobenius_error = 0.0;
        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < m; ++i) {
                frobenius_error += std::pow(A[i + j * m] - C[i + j * m], 2.0);
            }
        }
        frobenius_error = std::sqrt(frobenius_error);
        ASSERT_LE(frobenius_error, absolute_tolerance);
    }

    // Test thin SVD
    {
        std::vector<double> B = A;

        std::vector<double> U;
        std::vector<double> s;
        std::vector<double> Vt;
        long r;
        TruncatedSvd(m, n, B, relative_tolerance, true, r, U, s, Vt);

        ASSERT_EQ(r, k);

        SvdFactorQualityTest(m, n, r, U, s, Vt);

        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < k; ++i) {
                Vt[i + j * r] *= s[i];
            }
        }

        std::vector<double> C(m * n);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   m, n, r, 1.0, U.data(), m, Vt.data(), r, 0.0, C.data(), m);

        double frobenius_error = 0.0;
        double frobenius_norm = 0.0;
        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < m; ++i) {
                frobenius_error += std::pow(A[i + j * m] - C[i + j * m], 2.0);
                frobenius_norm += std::pow(A[i + j * m], 2.0);
            }
        }
        frobenius_error = std::sqrt(frobenius_error);
        frobenius_norm = std::sqrt(frobenius_norm);
        ASSERT_LE(frobenius_error, relative_tolerance * frobenius_norm);
    }
}

TEST(utils, TruncatedSvd) {
    TruncatedSvdTest(16, 64, 8);
    TruncatedSvdTest(32, 32, 8);
    TruncatedSvdTest(64, 16, 8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
