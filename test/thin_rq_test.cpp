#include "thin_rq.hpp"

#include <gtest/gtest.h>

#include <blas.hh>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

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

TEST(ThinRq, Short) {
    ThinRqTest<float>(4, 64);
    ThinRqTest<double>(4, 64);
}

TEST(ThinRq, Square) {
    ThinRqTest<float>(16, 16);
    ThinRqTest<double>(16, 16);
}

TEST(ThinRq, Tall) {
    ThinRqTest<float>(64, 4);
    ThinRqTest<double>(64, 4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
