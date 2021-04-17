#include "TensorFact_TtTensor.hpp"

#include <gtest/gtest.h>

#include <cmath>

TensorFact::TtTensor SumOfIndicesTtTensor(const std::vector<int> &size) {
    const int ndim = size.size();
    std::vector<int> rank(ndim + 1);
    std::vector<int> offset(ndim + 1);

    offset[0] = 0;
    rank[0] = 1;
    for (int d = 0; d < ndim; ++d) {
        rank[d + 1] = (d < ndim - 1) ? 2 : 1;
        offset[d + 1] = offset[d] + rank[d] * size[d] * rank[d + 1];
    }
    std::vector<double> param(offset[ndim]);

    for (int d = 0; d < ndim; ++d) {
        for (int i = 0; i < size[d]; ++i) {
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

    return TensorFact::TtTensor(ndim, size, rank, param);
}

TEST(TtTensor, ConstructFromParamAndEntry) {
    const std::vector<int> size{5, 3, 6, 4};
    TensorFact::TtTensor tt_tensor = SumOfIndicesTtTensor(size);

    for (int l = 0; l < 4; ++l) {
        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 5; ++i) {
                    ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}), i + j + k + l,
                                1.0e-15);
                }
            }
        }
    }
}

TEST(TtTensor, FileIO) {
    {
        const std::vector<int> size{5, 3, 6, 4};
        const TensorFact::TtTensor tt_tensor = SumOfIndicesTtTensor(size);
        tt_tensor.WriteToFile("tt_tensor.txt");
    }

    {
        TensorFact::TtTensor tt_tensor;
        tt_tensor.ReadFromFile("tt_tensor.txt");

        for (int l = 0; l < 4; ++l) {
            for (int k = 0; k < 6; ++k) {
                for (int j = 0; j < 3; ++j) {
                    for (int i = 0; i < 5; ++i) {
                        ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}),
                                    i + j + k + l, 1.0e-15);
                    }
                }
            }
        }
    }
}

TEST(TtTensor, Addition) {
    TensorFact::TtTensor tt_tensor1 = 5.0 * SumOfIndicesTtTensor({5, 3, 6, 4});
    TensorFact::TtTensor tt_tensor2 = -2.0 * SumOfIndicesTtTensor({5, 3, 6, 4});

    TensorFact::TtTensor tt_tensor = tt_tensor1 + tt_tensor2;

    for (int l = 0; l < 4; ++l) {
        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 5; ++i) {
                    ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}),
                                3.0 * (i + j + k + l), 1.0e-15);
                }
            }
        }
    }
}

TEST(TtTensor, ScalarMultiplication) {
    TensorFact::TtTensor tt_tensor1 = SumOfIndicesTtTensor({5, 3, 6, 4});
    TensorFact::TtTensor tt_tensor2 = 2.0 * tt_tensor1;

    for (int l = 0; l < 4; ++l) {
        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 5; ++i) {
                    ASSERT_NEAR(tt_tensor2.Entry({i, j, k, l}),
                                2.0 * (i + j + k + l), 1.0e-15);
                }
            }
        }
    }
}

TEST(TtTensor, ComputeFromFull) {
    const std::vector<int> size{3, 4, 5, 6};
    std::vector<double> array(360);
    for (int l = 0; l < 6; ++l) {
        for (int k = 0; k < 5; ++k) {
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 3; ++i) {
                    array[i + 3 * j + 12 * k + 60 * l] = i + j + k + l;
                }
            }
        }
    }

    float relative_tolerance = 1.0e-15;

    TensorFact::TtTensor tt_tensor;
    tt_tensor.ComputeFromFull(size, array, relative_tolerance);

    ASSERT_EQ(tt_tensor.Rank(0), 1);
    ASSERT_EQ(tt_tensor.Rank(1), 2);
    ASSERT_EQ(tt_tensor.Rank(2), 2);
    ASSERT_EQ(tt_tensor.Rank(3), 2);
    ASSERT_EQ(tt_tensor.Rank(4), 1);

    double frobenius_norm = 0.0;
    double absolute_error = 0.0;
    for (int l = 0; l < 6; ++l) {
        for (int k = 0; k < 5; ++k) {
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 3; ++i) {
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

TEST(TtTensor, Round) {
    const TensorFact::TtTensor tt_tensor = SumOfIndicesTtTensor({5, 3, 6, 4});

    const TensorFact::TtTensor tt_tensor_1 = 2.0 * tt_tensor;

    TensorFact::TtTensor tt_tensor_2 = tt_tensor + tt_tensor;
    tt_tensor_2.Round(1.0e-14);

    for (int l = 0; l < 4; ++l) {
        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 5; ++i) {
                    ASSERT_NEAR(tt_tensor_1.Entry({i, j, k, l}),
                                tt_tensor_2.Entry({i, j, k, l}), 1.0e-13);
                }
            }
        }
    }

    for (int d = 0; d <= 4; ++d) {
        ASSERT_EQ(tt_tensor_1.Rank(d), tt_tensor_2.Rank(d));
    }
}

TEST(TtTensor, Concatenate) {
    const TensorFact::TtTensor tt_tensor_1 = SumOfIndicesTtTensor({5, 3, 6, 4});
    const TensorFact::TtTensor tt_tensor_2 = SumOfIndicesTtTensor({5, 7, 6, 4});

    const TensorFact::TtTensor tt_tensor =
        tt_tensor_1.Concatenate(tt_tensor_2, 1, 1.0e-14);

    for (int l = 0; l < 4; ++l) {
        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < 10; ++j) {
                for (int i = 0; i < 5; ++i) {
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
    TensorFact::TtTensor tt_tensor1 = SumOfIndicesTtTensor({5, 3, 6, 4});
    TensorFact::TtTensor tt_tensor2 = -2.0 * SumOfIndicesTtTensor({5, 3, 6, 4});

    double obtained_value = tt_tensor1.Dot(tt_tensor2);

    double expected_value = 0.0;
    for (int l = 0; l < 4; ++l) {
        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 5; ++i) {
                    expected_value += std::pow(i + j + k + l, 2.0);
                }
            }
        }
    }
    expected_value *= -2.0;

    ASSERT_NEAR(obtained_value, expected_value, 1.0e-15);
}

TEST(TtTensor, FrobeniusNorm) {
    TensorFact::TtTensor tt_tensor = SumOfIndicesTtTensor({5, 3, 6, 4});

    float obtained_value = tt_tensor.FrobeniusNorm();

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

    ASSERT_TRUE(std::abs(obtained_value - expected_value) < 1.0e-05f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
