#include "tensorfact/tt_svd.hpp"

#include <gtest/gtest.h>

#include <cmath>

template <typename Real>
tensorfact::TtTensor<Real> SumOfIndicesTtTensor(const std::vector<long> &size) {
    const long ndim = size.size();
    std::vector<long> rank(ndim + 1);
    std::vector<long> offset(ndim + 1);

    offset[0] = 0;
    rank[0] = 1;
    for (long d = 0; d < ndim; ++d) {
        rank[d + 1] = (d < ndim - 1) ? 2 : 1;
        offset[d + 1] = offset[d] + rank[d] * size[d] * rank[d + 1];
    }
    std::vector<Real> param(offset[ndim]);

    for (long d = 0; d < ndim; ++d) {
        for (long i = 0; i < size[d]; ++i) {
            if (d == 0) {
                // (0, i, 0, d) = i
                param[i + offset[d]] = i;

                // (0, i, 1, d) = 1
                param[i + size[d] + offset[d]] = static_cast<Real>(1);
            } else if (d < ndim - 1) {
                // (0, i, 0, d) = 1
                param[2 * i + offset[d]] = static_cast<Real>(1);

                // (1, i, 0, d) = i
                param[1 + 2 * i + offset[d]] = i;

                // (0, i, 1, d) = 0
                param[2 * i + 2 * size[d] + offset[d]] = static_cast<Real>(0);

                // (1, i, 1, d) = 1
                param[1 + 2 * i + 2 * size[d] + offset[d]] =
                    static_cast<Real>(1);
            } else {
                // (0, i, 0, d) = 1
                param[2 * i + offset[d]] = static_cast<Real>(1);

                // (1, i, 0, d) = i
                param[1 + 2 * i + offset[d]] = i;
            }
        }
    }

    return tensorfact::TtTensor<Real>(ndim, size, rank, param);
}

TEST(TtSvd, TtSvdRelativeTolerance) {
    const std::vector<long> size{3, 4, 5, 6};
    std::vector<double> array(360);
    for (long l = 0; l < 6; ++l) {
        for (long k = 0; k < 5; ++k) {
            for (long j = 0; j < 4; ++j) {
                for (long i = 0; i < 3; ++i) {
                    array[i + 3 * j + 12 * k + 60 * l] = i + j + k + l;
                }
            }
        }
    }

    float relative_tolerance = 1.0e-15;

    tensorfact::TtTensor<double> tt_tensor =
        tensorfact::TtSvd<double>(size, array, relative_tolerance);

    const auto &rank = tt_tensor.Rank();
    ASSERT_EQ(rank[0], 1);
    ASSERT_EQ(rank[1], 2);
    ASSERT_EQ(rank[2], 2);
    ASSERT_EQ(rank[3], 2);
    ASSERT_EQ(rank[4], 1);

    double frobenius_norm = 0.0;
    double absolute_error = 0.0;
    for (long l = 0; l < 6; ++l) {
        for (long k = 0; k < 5; ++k) {
            for (long j = 0; j < 4; ++j) {
                for (long i = 0; i < 3; ++i) {
                    frobenius_norm += std::pow(i + j + k + l, 2.0);
                    absolute_error += std::pow(
                        i + j + k + l - tt_tensor.Entry({i, j, k, l}), 2.0);
                }
            }
        }
    }
    frobenius_norm = std::sqrt(frobenius_norm);
    absolute_error = std::sqrt(absolute_error);

    ASSERT_LE(absolute_error, relative_tolerance * frobenius_norm);
}

TEST(TtSvd, TtSvdMaxRank) {
    const std::vector<long> size{3, 4, 5, 6};
    std::vector<float> array(360);
    for (long l = 0; l < 6; ++l) {
        for (long k = 0; k < 5; ++k) {
            for (long j = 0; j < 4; ++j) {
                for (long i = 0; i < 3; ++i) {
                    array[i + 3 * j + 12 * k + 60 * l] = i + j + k + l;
                }
            }
        }
    }

    long max_rank = 2;

    tensorfact::TtTensor<float> tt_tensor =
        tensorfact::TtSvd<float>(size, array, max_rank);

    const auto &rank = tt_tensor.Rank();
    ASSERT_EQ(rank[0], 1);
    ASSERT_EQ(rank[1], 2);
    ASSERT_EQ(rank[2], 2);
    ASSERT_EQ(rank[3], 2);
    ASSERT_EQ(rank[4], 1);

    float frobenius_norm = 0.0f;
    float absolute_error = 0.0f;
    for (long l = 0; l < 6; ++l) {
        for (long k = 0; k < 5; ++k) {
            for (long j = 0; j < 4; ++j) {
                for (long i = 0; i < 3; ++i) {
                    frobenius_norm += std::pow(i + j + k + l, 2.0f);
                    absolute_error += std::pow(
                        i + j + k + l - tt_tensor.Entry({i, j, k, l}), 2.0f);
                }
            }
        }
    }
    frobenius_norm = std::sqrt(frobenius_norm);
    absolute_error = std::sqrt(absolute_error);

    ASSERT_NEAR(absolute_error / frobenius_norm, 0.0f, 1.0e-05f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
