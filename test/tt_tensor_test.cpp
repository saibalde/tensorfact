#include "tensorfact/tt_tensor.hpp"

#include <gtest/gtest.h>

#include <cmath>

tensorfact::TtTensor SumOfIndicesTtTensor(const std::vector<long> &size) {
    const long ndim = size.size();
    std::vector<long> rank(ndim + 1);
    std::vector<long> offset(ndim + 1);

    offset[0] = 0;
    rank[0] = 1;
    for (long d = 0; d < ndim; ++d) {
        rank[d + 1] = (d < ndim - 1) ? 2 : 1;
        offset[d + 1] = offset[d] + rank[d] * size[d] * rank[d + 1];
    }
    std::vector<double> param(offset[ndim]);

    for (long d = 0; d < ndim; ++d) {
        for (long i = 0; i < size[d]; ++i) {
            if (d == 0) {
                // (0, i, 0, d) = i
                param[i + offset[d]] = i;

                // (0, i, 1, d) = 1
                param[i + size[d] + offset[d]] = 1.0;
            } else if (d < ndim - 1) {
                // (0, i, 0, d) = 1
                param[2 * i + offset[d]] = 1.0;

                // (1, i, 0, d) = i
                param[1 + 2 * i + offset[d]] = i;

                // (0, i, 1, d) = 0
                param[2 * i + 2 * size[d] + offset[d]] = 0.0;

                // (1, i, 1, d) = 1
                param[1 + 2 * i + 2 * size[d] + offset[d]] = 1.0;
            } else {
                // (0, i, 0, d) = 1
                param[2 * i + offset[d]] = 1.0;

                // (1, i, 0, d) = i
                param[1 + 2 * i + offset[d]] = i;
            }
        }
    }

    return tensorfact::TtTensor(ndim, size, rank, param);
}

TEST(TtTensor, ConstructFromParamAndEntry) {
    const std::vector<long> size{5, 3, 6, 4};
    tensorfact::TtTensor tt_tensor = SumOfIndicesTtTensor(size);

    for (long l = 0; l < 4; ++l) {
        for (long k = 0; k < 6; ++k) {
            for (long j = 0; j < 3; ++j) {
                for (long i = 0; i < 5; ++i) {
                    ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}), i + j + k + l,
                                1.0e-15);
                }
            }
        }
    }
}

TEST(TtTensor, FileIO) {
    {
        const std::vector<long> size{5, 3, 6, 4};
        const tensorfact::TtTensor tt_tensor = SumOfIndicesTtTensor(size);
        tt_tensor.WriteToFile("tt_tensor.txt");
    }

    {
        tensorfact::TtTensor tt_tensor;
        tt_tensor.ReadFromFile("tt_tensor.txt");

        for (long l = 0; l < 4; ++l) {
            for (long k = 0; k < 6; ++k) {
                for (long j = 0; j < 3; ++j) {
                    for (long i = 0; i < 5; ++i) {
                        ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}),
                                    i + j + k + l, 1.0e-15);
                    }
                }
            }
        }
    }
}

TEST(TtTensor, Addition) {
    tensorfact::TtTensor tt_tensor1 = 5.0 * SumOfIndicesTtTensor({5, 3, 6, 4});
    tensorfact::TtTensor tt_tensor2 = -2.0 * SumOfIndicesTtTensor({5, 3, 6, 4});

    tensorfact::TtTensor tt_tensor = tt_tensor1 + tt_tensor2;

    for (long l = 0; l < 4; ++l) {
        for (long k = 0; k < 6; ++k) {
            for (long j = 0; j < 3; ++j) {
                for (long i = 0; i < 5; ++i) {
                    ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}),
                                3.0 * (i + j + k + l), 1.0e-15);
                }
            }
        }
    }
}

TEST(TtTensor, ScalarMultiplication) {
    tensorfact::TtTensor tt_tensor1 = SumOfIndicesTtTensor({5, 3, 6, 4});
    tensorfact::TtTensor tt_tensor2 = 2.0 * tt_tensor1;

    for (long l = 0; l < 4; ++l) {
        for (long k = 0; k < 6; ++k) {
            for (long j = 0; j < 3; ++j) {
                for (long i = 0; i < 5; ++i) {
                    ASSERT_NEAR(tt_tensor2.Entry({i, j, k, l}),
                                2.0 * (i + j + k + l), 1.0e-15);
                }
            }
        }
    }
}

TEST(TtTensor, Round) {
    const tensorfact::TtTensor tt_tensor = SumOfIndicesTtTensor({5, 3, 6, 4});

    const tensorfact::TtTensor tt_tensor_1 = 2.0 * tt_tensor;

    tensorfact::TtTensor tt_tensor_2 = tt_tensor + tt_tensor;
    tt_tensor_2.Round(1.0e-14);

    for (long l = 0; l < 4; ++l) {
        for (long k = 0; k < 6; ++k) {
            for (long j = 0; j < 3; ++j) {
                for (long i = 0; i < 5; ++i) {
                    ASSERT_NEAR(tt_tensor_1.Entry({i, j, k, l}),
                                tt_tensor_2.Entry({i, j, k, l}), 1.0e-13);
                }
            }
        }
    }

    for (long d = 0; d <= 4; ++d) {
        ASSERT_EQ(tt_tensor_1.Rank(d), tt_tensor_2.Rank(d));
    }
}

TEST(TtTensor, Concatenate) {
    const tensorfact::TtTensor tt_tensor_1 = SumOfIndicesTtTensor({5, 3, 6, 4});
    const tensorfact::TtTensor tt_tensor_2 = SumOfIndicesTtTensor({5, 7, 6, 4});

    const tensorfact::TtTensor tt_tensor =
        tt_tensor_1.Concatenate(tt_tensor_2, 1, 1.0e-14);

    for (long l = 0; l < 4; ++l) {
        for (long k = 0; k < 6; ++k) {
            for (long j = 0; j < 10; ++j) {
                for (long i = 0; i < 5; ++i) {
                    if (j < 3) {
                        ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}),
                                    tt_tensor_1.Entry({i, j, k, l}), 1.0e-13);
                    } else {
                        ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}),
                                    tt_tensor_2.Entry({i, j - 3, k, l}),
                                    1.0e-13);
                    }
                }
            }
        }
    }
}

TEST(TtTensor, DotProduct) {
    tensorfact::TtTensor tt_tensor1 = SumOfIndicesTtTensor({5, 3, 6, 4});
    tensorfact::TtTensor tt_tensor2 = -2.0 * SumOfIndicesTtTensor({5, 3, 6, 4});

    double obtained_value = tt_tensor1.Dot(tt_tensor2);

    double expected_value = 0.0;
    for (long l = 0; l < 4; ++l) {
        for (long k = 0; k < 6; ++k) {
            for (long j = 0; j < 3; ++j) {
                for (long i = 0; i < 5; ++i) {
                    expected_value += std::pow(i + j + k + l, 2.0);
                }
            }
        }
    }
    expected_value *= -2.0;

    ASSERT_NEAR(obtained_value, expected_value, 1.0e-15);
}

TEST(TtTensor, FrobeniusNorm) {
    tensorfact::TtTensor tt_tensor = SumOfIndicesTtTensor({5, 3, 6, 4});

    float obtained_value = tt_tensor.FrobeniusNorm();

    float expected_value = 0.0f;
    for (long l = 0; l < 4; ++l) {
        for (long k = 0; k < 6; ++k) {
            for (long j = 0; j < 3; ++j) {
                for (long i = 0; i < 5; ++i) {
                    expected_value += std::pow(i + j + k + l, 2.0f);
                }
            }
        }
    }
    expected_value = std::sqrt(expected_value);

    ASSERT_TRUE(std::abs(obtained_value - expected_value) < 1.0e-05f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
