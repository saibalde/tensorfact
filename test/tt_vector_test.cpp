#include "tt_vector.hpp"

#include <armadillo>
#include <cmath>
#include <limits>

#include "gtest/gtest.h"

/// Check if floating point values are approximately equal
template <typename Real>
bool isApproxEqual(Real a, Real b,
                   Real eps = 10 * std::numeric_limits<Real>::epsilon()) {
    return std::abs(a - b) < eps;
}

/// Create TT-Vector \f$v(i_0, ..., i_{d - 1}) = i_0 + ... + i_{d - 1}\f$
template <typename Real, typename Index>
TtVector<Real, Index> createTestTtVector(const arma::Col<Index> &dims) {
    constexpr Index ZERO = static_cast<Index>(0);
    constexpr Index ONE = static_cast<Index>(1);
    constexpr Index TWO = static_cast<Index>(2);

    Index ndim = dims.n_elem;

    arma::Col<Index> ranks(ndim + ONE);
    ranks(ZERO) = ONE;
    for (Index d = ONE; d < ndim; ++d) {
        ranks(d) = TWO;
    }
    ranks(ndim) = ONE;

    arma::field<arma::Cube<Real>> cores(ndim);

    cores(ZERO).zeros(ONE, TWO, dims(ZERO));
    for (Index i = ZERO; i < dims(ZERO); ++i) {
        cores(ZERO)(ZERO, ZERO, i) = i;
        cores(ZERO)(ZERO, ONE, i) = ONE;
    }

    for (Index d = ONE; d < ndim - ONE; ++d) {
        cores(d).zeros(TWO, TWO, dims(d));
        for (Index i = ZERO; i < dims(d); ++i) {
            cores(d)(ZERO, ZERO, i) = ONE;
            cores(d)(ONE, ZERO, i) = i;
            cores(d)(ZERO, ONE, i) = ZERO;
            cores(d)(ONE, ONE, i) = ONE;
        }
    }

    cores(ndim - ONE).set_size(TWO, ONE, dims(ndim - ONE));
    for (Index i = ZERO; i < dims(ndim - ONE); ++i) {
        cores(ndim - ONE)(ZERO, ZERO, i) = ONE;
        cores(ndim - ONE)(ONE, ZERO, i) = i;
    }

    return TtVector<Real, Index>(cores);
}

TEST(vector, construct_core) {
    TtVector<float, int> tt_vector =
        createTestTtVector<float, int>({5, 3, 6, 4});

    ASSERT_TRUE(tt_vector.ndim() == 4);
    ASSERT_TRUE(arma::all(tt_vector.size() == arma::Col<int>({5, 3, 6, 4})));
    ASSERT_TRUE(
        arma::all(tt_vector.ranks() == arma::Col<int>({1, 2, 2, 2, 1})));
}

TEST(vector, construct_svd) {
    arma::Col<int> size{3, 4, 5, 6};
    int numel = arma::prod(size);
    arma::Col<float> array(numel);

    for (int l = 0; l < 6; ++l) {
        for (int k = 0; k < 5; ++k) {
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 3; ++i) {
                    array(i + 3 * j + 12 * k + 60 * l) = i + j + k + l;
                }
            }
        }
    }

    TtVector<float, int> tt_vector(array, size, 1.0e-04f);

    auto ranks = tt_vector.ranks();
    ASSERT_EQ(ranks(0), 1);
    ASSERT_EQ(ranks(1), 2);
    ASSERT_EQ(ranks(2), 2);
    ASSERT_EQ(ranks(3), 2);
    ASSERT_EQ(ranks(4), 1);

    for (int l = 0; l < 6; ++l) {
        for (int k = 0; k < 5; ++k) {
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 3; ++i) {
                    ASSERT_TRUE(isApproxEqual<float>(
                        tt_vector({i, j, k, l}), i + j + k + l,
                        1000 * std::numeric_limits<float>::epsilon()));
                }
            }
        }
    }
}

TEST(vector, vector_addition) {
    TtVector<float, int> tt_vector1 =
        5.0f * createTestTtVector<float, int>({5, 3, 6, 4});
    TtVector<float, int> tt_vector2 =
        -2.0f * createTestTtVector<float, int>({5, 3, 6, 4});

    TtVector<float, int> tt_vector = tt_vector1 + tt_vector2;

    for (int l = 0; l < 4; ++l) {
        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 5; ++i) {
                    ASSERT_TRUE(isApproxEqual<float>(tt_vector({i, j, k, l}),
                                                     3.0f * (i + j + k + l)));
                }
            }
        }
    }
}

TEST(vector, scalar_multiplication) {
    TtVector<float, int> tt_vector1 =
        createTestTtVector<float, int>({5, 3, 6, 4});
    TtVector<float, int> tt_vector2 = 2.0f * tt_vector1;

    for (int l = 0; l < 4; ++l) {
        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 5; ++i) {
                    ASSERT_TRUE(isApproxEqual<float>(tt_vector2({i, j, k, l}),
                                                     2.0f * (i + j + k + l)));
                }
            }
        }
    }
}

TEST(vector, dot_product) {
    TtVector<float, int> tt_vector1 =
        createTestTtVector<float, int>({5, 3, 6, 4});
    TtVector<float, int> tt_vector2 =
        -2.0f * createTestTtVector<float, int>({5, 3, 6, 4});

    float obtained_value = dot(tt_vector1, tt_vector2);

    float expected_value = 0.0f;
    for (int l = 0; l < 4; ++l) {
        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 5; ++i) {
                    expected_value += std::pow(i + j + k + l, 2.0f);
                }
            }
        }
    }
    expected_value *= -2.0f;

    ASSERT_TRUE(isApproxEqual<float>(obtained_value, expected_value));
}

TEST(vector, vector_norm) {
    TtVector<float, int> tt_vector =
        createTestTtVector<float, int>({5, 3, 6, 4});

    float obtained_value = norm2(tt_vector);

    float expected_value = 0.0f;
    for (int l = 0; l < 4; ++l) {
        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 5; ++i) {
                    expected_value += std::pow(i + j + k + l, 2.0f);
                }
            }
        }
    }
    expected_value = std::sqrt(expected_value);

    ASSERT_TRUE(isApproxEqual<float>(obtained_value, expected_value));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
