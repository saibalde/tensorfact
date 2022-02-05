#include "truncated_svd.hpp"

#include <gtest/gtest.h>

#include <armadillo>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "thin_lq.hpp"

template <typename Real>
void CreateTestMatrix(long m, long n, long r, arma::Mat<Real> &A,
                      long &rank_cutoff, Real &absolute_tolerance,
                      Real &relative_tolerance) {
    if (m < 1 || n < 1) {
        throw std::invalid_argument(
            "Number of rows and columns must be positive");
    }

    // Create Ut factor
    arma::Mat<Real> Ut;
    {
        arma::Mat<Real> temp(r, m, arma::fill::randn);

        arma::Mat<Real> L;
        ThinLq<Real>(temp, L, Ut);
    }

    // Create s factor; s[i] ~ Uniform([(r - 1 - i), (r - i)])
    arma::Col<Real> s(r, arma::fill::randu);
    for (long i = 0; i < r; ++i) {
        s(i) += r - i - 1;
    }

    // Create Vt factor
    arma::Mat<Real> Vt;
    {
        arma::Mat<Real> temp(r, n, arma::fill::randn);

        arma::Mat<Real> L;
        ThinLq<Real>(temp, L, Vt);
    }

    // Compute A = Ut.conjugate() * diagmat(s) * Vt
    A = Ut.t() * arma::diagmat(s) * Vt;

    // Compute absolute and relative errors given specified maximum rank
    std::vector<Real> absolute_error(r);

    absolute_error[r - 1] = static_cast<Real>(0);
    for (long i = r - 1; i > 0; --i) {
        absolute_error[i - 1] = absolute_error[i] + std::pow(s(i), 2);
    }
    Real frobenius_norm = absolute_error[0] + std::pow(s(0), 2);

    for (long i = 0; i < r; ++i) {
        absolute_error[i] = std::sqrt(absolute_error[i]);
    }
    frobenius_norm = std::sqrt(frobenius_norm);

    std::vector<Real> relative_error(r);

    for (long i = 0; i < r; ++i) {
        relative_error[i] = absolute_error[i] / frobenius_norm;
    }

    // Determine 75% Frobenius norm cutoff rank
    rank_cutoff = 1;
    while (rank_cutoff <= r) {
        if (relative_error[rank_cutoff - 1] <= static_cast<Real>(0.25)) {
            break;
        }
        ++rank_cutoff;
    }

    // Compute accuracy levels based on cutoff rank
    absolute_tolerance =
        (absolute_error[rank_cutoff - 2] + absolute_error[rank_cutoff - 1]) / 2;
    relative_tolerance =
        (relative_error[rank_cutoff - 2] + relative_error[rank_cutoff - 1]) / 2;
}

template <typename Real>
void SvdFactorQualityTest(const arma::Mat<Real> &U, const arma::Col<Real> &s,
                          const arma::Mat<Real> &V) {
    const long m = U.n_rows;
    const long n = V.n_rows;
    const long r = s.n_rows;

    for (long j1 = 0; j1 < r; ++j1) {
        for (long j2 = 0; j2 < j1; ++j2) {
            Real dot_product = static_cast<Real>(0);
            for (long i = 0; i < m; ++i) {
                dot_product += U(i, j1) * U(i, j2);
            }

            ASSERT_NEAR(dot_product, static_cast<Real>(0),
                        100 * std::numeric_limits<Real>::epsilon());
        }

        Real norm_squared = static_cast<Real>(0);
        for (long i = 0; i < m; ++i) {
            norm_squared += std::pow(U(i, j1), 2);
        }

        ASSERT_NEAR(norm_squared, static_cast<Real>(1),
                    100 * std::numeric_limits<Real>::epsilon());
    }

    for (long i = 0; i < r; ++i) {
        if (i < r - 1) {
            ASSERT_GE(s(i), s(i + 1));
        } else {
            ASSERT_GT(s(i), static_cast<Real>(0));
        }
    }

    for (long j1 = 0; j1 < r; ++j1) {
        for (long j2 = 0; j2 < j1; ++j2) {
            Real dot_product = static_cast<Real>(0);
            for (long i = 0; i < n; ++i) {
                dot_product += V(i, j1) * V(i, j2);
            }

            ASSERT_NEAR(dot_product, static_cast<Real>(0),
                        100 * std::numeric_limits<Real>::epsilon());
        }

        Real norm_squared = static_cast<Real>(0);
        for (long i = 0; i < n; ++i) {
            norm_squared += std::pow(V(i, j1), 2);
        }

        ASSERT_NEAR(norm_squared, static_cast<Real>(1),
                    100 * std::numeric_limits<Real>::epsilon());
    }
}

template <typename Real>
void ThinSvdTest(long m, long n, long r) {
    if (m < 1 || n < 1 || r < 1) {
        throw std::invalid_argument("Matrix sizes and rank must be positive");
    }

    if (r > m || r > n) {
        throw std::invalid_argument("Rank must be smaller than matrix size");
    }

    // Construct appropriate matrix
    arma::Mat<Real> A;
    long rank_cutoff;
    Real absolute_tolerance;
    Real relative_tolerance;
    CreateTestMatrix<Real>(m, n, r, A, rank_cutoff, absolute_tolerance,
                           relative_tolerance);

    // Test
    arma::Mat<Real> U;
    arma::Col<Real> s;
    arma::Mat<Real> V;
    long rank;
    TruncatedSvd<Real>(A, 0, false, U, s, V, rank);

    const Real frobenius_error =
        arma::norm(A - U * arma::diagmat(s) * V.t(), "fro");

    ASSERT_EQ(rank, std::min(m, n));
    SvdFactorQualityTest<Real>(U, s, V);
    ASSERT_LE(frobenius_error, 1000 * std::numeric_limits<Real>::epsilon());
}

template <typename Real>
void TruncatedSvdAbsoluteToleranceTest(long m, long n, long r) {
    if (m < 1 || n < 1 || r < 1) {
        throw std::invalid_argument("Matrix sizes and rank must be positive");
    }

    if (r > m || r > n) {
        throw std::invalid_argument("Rank must be smaller than matrix size");
    }

    // Construct appropriate matrix
    arma::Mat<Real> A;
    long rank_cutoff;
    Real absolute_tolerance;
    Real relative_tolerance;
    CreateTestMatrix<Real>(m, n, r, A, rank_cutoff, absolute_tolerance,
                           relative_tolerance);

    // Test
    arma::Mat<Real> U;
    arma::Col<Real> s;
    arma::Mat<Real> V;
    long rank;
    TruncatedSvd<Real>(A, absolute_tolerance, false, U, s, V, rank);

    const Real frobenius_error =
        arma::norm(A - U * arma::diagmat(s) * V.t(), "fro");

    ASSERT_EQ(rank, rank_cutoff);
    SvdFactorQualityTest<Real>(U, s, V);
    ASSERT_LE(frobenius_error, absolute_tolerance);
}

template <typename Real>
void TruncatedSvdRelativeToleranceTest(long m, long n, long r) {
    if (m < 1 || n < 1 || r < 1) {
        throw std::invalid_argument("Matrix sizes and rank must be positive");
    }

    if (r > m || r > n) {
        throw std::invalid_argument("Rank must be smaller than matrix size");
    }

    // Construct appropriate matrix
    arma::Mat<Real> A;
    long rank_cutoff;
    Real absolute_tolerance;
    Real relative_tolerance;
    CreateTestMatrix<Real>(m, n, r, A, rank_cutoff, absolute_tolerance,
                           relative_tolerance);

    // Test
    arma::Mat<Real> U;
    arma::Col<Real> s;
    arma::Mat<Real> V;
    long rank;
    TruncatedSvd<Real>(A, relative_tolerance, true, U, s, V, rank);

    const Real frobenius_norm = arma::norm(A, "fro");
    const Real frobenius_error =
        arma::norm(A - U * arma::diagmat(s) * V.t(), "fro");

    ASSERT_EQ(rank, rank_cutoff);
    SvdFactorQualityTest<Real>(U, s, V);
    ASSERT_LE(frobenius_error, relative_tolerance * frobenius_norm);
}

TEST(TruncatedSvd, ThinSvd_ShortMatrix) {
    ThinSvdTest<float>(16, 64, 8);
    ThinSvdTest<double>(16, 64, 8);
}

TEST(TruncatedSvd, ThinSvd_SquareMatrix) {
    ThinSvdTest<float>(32, 32, 8);
    ThinSvdTest<double>(32, 32, 8);
}

TEST(TruncatedSvd, ThinSvd_TallMatrix) {
    ThinSvdTest<float>(64, 16, 8);
    ThinSvdTest<double>(64, 16, 8);
}

TEST(TruncatedSvd, TruncatedSvdAbsoluteTolerance_ShortMatrix) {
    TruncatedSvdAbsoluteToleranceTest<float>(16, 64, 8);
    TruncatedSvdAbsoluteToleranceTest<double>(16, 64, 8);
}

TEST(TruncatedSvd, TruncatedSvdAbsoluteTolerance_SquareMatrix) {
    TruncatedSvdAbsoluteToleranceTest<float>(32, 32, 8);
    TruncatedSvdAbsoluteToleranceTest<double>(32, 32, 8);
}

TEST(TruncatedSvd, TruncatedSvdAbsoluteTolerance_TallMatrix) {
    TruncatedSvdAbsoluteToleranceTest<float>(64, 16, 8);
    TruncatedSvdAbsoluteToleranceTest<double>(64, 16, 8);
}

TEST(TruncatedSvd, TruncatedSvdRelativeTolerance_ShortMatrix) {
    TruncatedSvdRelativeToleranceTest<float>(16, 64, 8);
    TruncatedSvdRelativeToleranceTest<double>(16, 64, 8);
}

TEST(TruncatedSvd, TruncatedSvdRelativeTolerance_SquareMatrix) {
    TruncatedSvdRelativeToleranceTest<float>(32, 32, 8);
    TruncatedSvdRelativeToleranceTest<double>(32, 32, 8);
}

TEST(TruncatedSvd, TruncatedSvdRelativeTolerance_TallMatrix) {
    TruncatedSvdRelativeToleranceTest<float>(64, 16, 8);
    TruncatedSvdRelativeToleranceTest<double>(64, 16, 8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
