#include "tensorfact/tt_tensor.hpp"

#include <armadillo>
#include <cmath>
#include <limits>

#include "gtest/gtest.h"

/// Check if floating point values are approximately equal
template <typename Real>
bool IsApproximatelyEqual(Real a, Real b,
                          Real eps = 10 *
                                     std::numeric_limits<Real>::epsilon()) {
    return std::abs(a - b) < eps;
}

/// Create TT-Vector \f$v(i_0, ..., i_{d - 1}) = i_0 + ... + i_{d - 1}\f$
template <typename Real>
tensorfact::TtTensor<Real> CreateTestTtTensor(
    const arma::Col<arma::uword> &dims) {
    arma::uword ndim = dims.n_elem;

    arma::Col<arma::uword> ranks(ndim + 1);
    ranks(0) = 1;
    for (arma::uword d = 1; d < ndim; ++d) {
        ranks(d) = 2;
    }
    ranks(ndim) = 1;

    arma::field<arma::Cube<Real>> cores(ndim);

    cores(0).zeros(1, dims(0), 2);
    for (arma::uword i = 0; i < dims(0); ++i) {
        cores(0)(0, i, 0) = i;
        cores(0)(0, i, 1) = 1;
    }

    for (arma::uword d = 1; d < ndim - 1; ++d) {
        cores(d).zeros(2, dims(d), 2);
        for (arma::uword i = 0; i < dims(d); ++i) {
            cores(d)(0, i, 0) = 1;
            cores(d)(1, i, 0) = i;
            cores(d)(0, i, 1) = 0;
            cores(d)(1, i, 1) = 1;
        }
    }

    cores(ndim - 1).set_size(2, dims(ndim - 1), 1);
    for (arma::uword i = 0; i < dims(ndim - 1); ++i) {
        cores(ndim - 1)(0, i, 0) = 1;
        cores(ndim - 1)(1, i, 0) = i;
    }

    return tensorfact::TtTensor<Real>(cores);
}

TEST(tt_tensor, ConstructFromCore) {
    tensorfact::TtTensor<float> tt_tensor =
        CreateTestTtTensor<float>({5, 3, 6, 4});

    ASSERT_TRUE(tt_tensor.NDim() == 4);
    ASSERT_TRUE(
        arma::all(tt_tensor.Size() == arma::Col<arma::uword>({5, 3, 6, 4})));
    ASSERT_TRUE(
        arma::all(tt_tensor.Rank() == arma::Col<arma::uword>({1, 2, 2, 2, 1})));
}

TEST(tt_tensor, ConstructWithSvd) {
    arma::Col<arma::uword> size{3, 4, 5, 6};
    arma::uword numel = arma::prod(size);
    arma::Col<float> array(numel);

    for (arma::uword l = 0; l < 6; ++l) {
        for (arma::uword k = 0; k < 5; ++k) {
            for (arma::uword j = 0; j < 4; ++j) {
                for (arma::uword i = 0; i < 3; ++i) {
                    array(i + 3 * j + 12 * k + 60 * l) = i + j + k + l;
                }
            }
        }
    }

    float rel_acc = 10 * std::numeric_limits<float>::epsilon();
    tensorfact::TtTensor<float> tt_tensor(array, size, rel_acc);

    auto ranks = tt_tensor.Rank();
    ASSERT_EQ(ranks(0), 1);
    ASSERT_EQ(ranks(1), 2);
    ASSERT_EQ(ranks(2), 2);
    ASSERT_EQ(ranks(3), 2);
    ASSERT_EQ(ranks(4), 1);

    float error_squared = 0.0f;
    for (arma::uword l = 0; l < 6; ++l) {
        for (arma::uword k = 0; k < 5; ++k) {
            for (arma::uword j = 0; j < 4; ++j) {
                for (arma::uword i = 0; i < 3; ++i) {
                    error_squared +=
                        std::pow(i + j + k + l - tt_tensor({i, j, k, l}), 2.0f);
                }
            }
        }
    }
    ASSERT_TRUE(std::sqrt(error_squared) <= rel_acc * arma::norm(array));
}

TEST(tt_tensor, Addition) {
    tensorfact::TtTensor<float> tt_tensor1 =
        5.0f * CreateTestTtTensor<float>({5, 3, 6, 4});
    tensorfact::TtTensor<float> tt_tensor2 =
        -2.0f * CreateTestTtTensor<float>({5, 3, 6, 4});

    tensorfact::TtTensor<float> tt_tensor = tt_tensor1 + tt_tensor2;

    for (arma::uword l = 0; l < 4; ++l) {
        for (arma::uword k = 0; k < 6; ++k) {
            for (arma::uword j = 0; j < 3; ++j) {
                for (arma::uword i = 0; i < 5; ++i) {
                    ASSERT_TRUE(IsApproximatelyEqual<float>(
                        tt_tensor({i, j, k, l}), 3.0f * (i + j + k + l)));
                }
            }
        }
    }
}

TEST(tt_tensor, ScalarMultiplication) {
    tensorfact::TtTensor<float> tt_tensor1 =
        CreateTestTtTensor<float>({5, 3, 6, 4});
    tensorfact::TtTensor<float> tt_tensor2 = 2.0f * tt_tensor1;

    for (arma::uword l = 0; l < 4; ++l) {
        for (arma::uword k = 0; k < 6; ++k) {
            for (arma::uword j = 0; j < 3; ++j) {
                for (arma::uword i = 0; i < 5; ++i) {
                    ASSERT_TRUE(IsApproximatelyEqual<float>(
                        tt_tensor2({i, j, k, l}), 2.0f * (i + j + k + l)));
                }
            }
        }
    }
}

TEST(tt_tensor, DotProduct) {
    tensorfact::TtTensor<float> tt_tensor1 =
        CreateTestTtTensor<float>({5, 3, 6, 4});
    tensorfact::TtTensor<float> tt_tensor2 =
        -2.0f * CreateTestTtTensor<float>({5, 3, 6, 4});

    float obtained_value = tt_tensor1.Dot(tt_tensor2);

    float expected_value = 0.0f;
    for (arma::uword l = 0; l < 4; ++l) {
        for (arma::uword k = 0; k < 6; ++k) {
            for (arma::uword j = 0; j < 3; ++j) {
                for (arma::uword i = 0; i < 5; ++i) {
                    expected_value += std::pow(i + j + k + l, 2.0f);
                }
            }
        }
    }
    expected_value *= -2.0f;

    ASSERT_TRUE(IsApproximatelyEqual<float>(obtained_value, expected_value));
}

TEST(tt_tensor, Norm) {
    tensorfact::TtTensor<float> tt_tensor =
        CreateTestTtTensor<float>({5, 3, 6, 4});

    float obtained_value = tt_tensor.Norm2();

    float expected_value = 0.0f;
    for (arma::uword l = 0; l < 4; ++l) {
        for (arma::uword k = 0; k < 6; ++k) {
            for (arma::uword j = 0; j < 3; ++j) {
                for (arma::uword i = 0; i < 5; ++i) {
                    expected_value += std::pow(i + j + k + l, 2.0f);
                }
            }
        }
    }
    expected_value = std::sqrt(expected_value);

    ASSERT_TRUE(IsApproximatelyEqual<float>(obtained_value, expected_value));
}

TEST(tt_tensor, Round) {
    const tensorfact::TtTensor<float> tt_tensor =
        CreateTestTtTensor<float>({5, 3, 6, 4});

    const tensorfact::TtTensor<float> tt_tensor_1 = 2.0f * tt_tensor;
    const tensorfact::TtTensor<float> tt_tensor_2 = tt_tensor + tt_tensor;
    const tensorfact::TtTensor<float> tt_tensor_3 = tt_tensor_2.Round(1.0e-06f);

    ASSERT_TRUE((tt_tensor_2 - tt_tensor_3).Norm2() / tt_tensor_2.Norm2() <
                5.0e-04f);

    const arma::uword &ndim = tt_tensor_1.NDim();

    const arma::Col<arma::uword> &rank_1 = tt_tensor_1.Rank();
    const arma::Col<arma::uword> &rank_3 = tt_tensor_3.Rank();

    for (arma::uword d = 0; d <= ndim; ++d) {
        ASSERT_TRUE(rank_1(d) == rank_3(d));
    }
}

TEST(tt_tensor, AddZeroPaddingBack) {
    tensorfact::TtTensor<float> tt_tensor_1 =
        CreateTestTtTensor<float>({5, 3, 6, 4});
    tensorfact::TtTensor<float> tt_tensor_2 =
        tt_tensor_1.AddZeroPaddingBack(1, 4);

    for (arma::uword l = 0; l < 4; ++l) {
        for (arma::uword k = 0; k < 6; ++k) {
            for (arma::uword j = 0; j < 7; ++j) {
                for (arma::uword i = 0; i < 5; ++i) {
                    if (j < 3) {
                        ASSERT_TRUE(IsApproximatelyEqual<float>(
                            tt_tensor_1({i, j, k, l}),
                            tt_tensor_2({i, j, k, l})));
                    } else {
                        ASSERT_TRUE(IsApproximatelyEqual<float>(
                            0.0f, tt_tensor_2({i, j, k, l})));
                    }
                }
            }
        }
    }
}

TEST(tt_tensor, AddZeroPaddingFront) {
    tensorfact::TtTensor<float> tt_tensor_1 =
        CreateTestTtTensor<float>({5, 3, 6, 4});
    tensorfact::TtTensor<float> tt_tensor_2 =
        tt_tensor_1.AddZeroPaddingFront(2, 3);

    for (arma::uword l = 0; l < 4; ++l) {
        for (arma::uword k = 0; k < 9; ++k) {
            for (arma::uword j = 0; j < 3; ++j) {
                for (arma::uword i = 0; i < 5; ++i) {
                    if (k < 3) {
                        ASSERT_TRUE(IsApproximatelyEqual<float>(
                            0.0f, tt_tensor_2({i, j, k, l})));
                    } else {
                        ASSERT_TRUE(IsApproximatelyEqual<float>(
                            tt_tensor_1({i, j, k - 3, l}),
                            tt_tensor_2({i, j, k, l})));
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
