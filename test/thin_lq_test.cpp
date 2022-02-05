#include "thin_lq.hpp"

#include <gtest/gtest.h>

#include <armadillo>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

template <typename Real>
void ThinLqTest(long m, long n) {
    arma::Mat<Real> A(m, n, arma::fill::randn);

    arma::Mat<Real> L;
    arma::Mat<Real> Q;
    ThinLq<Real>(A, L, Q);

    const long k = std::min(m, n);

    ASSERT_EQ(L.size(), m * k);
    ASSERT_EQ(Q.size(), k * n);

    for (long j = 0; j < k; ++j) {
        for (long i = 0; i < j; ++i) {
            ASSERT_NEAR(L(i, j), static_cast<Real>(0),
                        10 * std::numeric_limits<Real>::epsilon());
        }
    }

    for (long i1 = 0; i1 < k; ++i1) {
        for (long i2 = 0; i2 < i1; ++i2) {
            Real dot_product = static_cast<Real>(0);
            for (long j = 0; j < n; ++j) {
                dot_product += Q(i1, j) * Q(i2, j);
            }

            ASSERT_NEAR(dot_product, static_cast<Real>(0),
                        10 * std::numeric_limits<Real>::epsilon());
        }

        Real norm_squared = static_cast<Real>(0);
        for (long j = 0; j < n; ++j) {
            norm_squared += std::pow(Q(i1, j), 2);
        }

        ASSERT_NEAR(norm_squared, static_cast<Real>(1),
                    10 * std::numeric_limits<Real>::epsilon());
    }

    arma::Mat<Real> B = L * Q;

    Real frobenius_error = static_cast<Real>(0);
    for (long j = 0; j < n; ++j) {
        for (long i = 0; i < m; ++i) {
            frobenius_error += std::pow(B(i, j) - A(i, j), 2);
        }
    }
    frobenius_error = std::sqrt(frobenius_error);
    ASSERT_NEAR(frobenius_error, static_cast<Real>(0),
                100 * std::numeric_limits<Real>::epsilon());
}

TEST(ThinLq, Short) {
    ThinLqTest<float>(4, 64);
    ThinLqTest<double>(4, 64);
}

TEST(ThinLq, Square) {
    ThinLqTest<float>(16, 16);
    ThinLqTest<double>(16, 16);
}

TEST(ThinLq, Tall) {
    ThinLqTest<float>(64, 4);
    ThinLqTest<double>(64, 4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
