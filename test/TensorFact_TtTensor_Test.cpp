#include "TensorFact_TtTensor.hpp"

#include <cmath>

#include "TensorFact_Array.hpp"
#include "gtest/gtest.h"

template <typename Scalar>
TensorFact::TtTensor<Scalar> SumOfIndicesTtTensor(
    const std::vector<std::size_t> &dims) {
    const std::size_t ndim = dims.size();

    std::vector<TensorFact::Array<Scalar>> cores(ndim);

    cores[0].Resize({1, dims[0], 2});
    for (std::size_t i = 0; i < dims[0]; ++i) {
        cores[0]({0, i, 0}) = static_cast<Scalar>(i);
        cores[0]({0, i, 1}) = static_cast<Scalar>(1);
    }

    for (std::size_t d = 1; d < ndim - 1; ++d) {
        cores[d].Resize({2, dims[d], 2});
        for (std::size_t i = 0; i < dims[d]; ++i) {
            cores[d]({0, i, 0}) = static_cast<Scalar>(1);
            cores[d]({1, i, 0}) = static_cast<Scalar>(i);
            cores[d]({0, i, 1}) = static_cast<Scalar>(0);
            cores[d]({1, i, 1}) = static_cast<Scalar>(1);
        }
    }

    cores[ndim - 1].Resize({2, dims[ndim - 1], 1});
    for (std::size_t i = 0; i < dims[ndim - 1]; ++i) {
        cores[ndim - 1]({0, i, 0}) = static_cast<Scalar>(1);
        cores[ndim - 1]({1, i, 0}) = static_cast<Scalar>(i);
    }

    return TensorFact::TtTensor<Scalar>(cores);
}

TEST(TtTensor, ConstructFromCore) {
    const std::vector<std::size_t> size{5, 3, 6, 4};
    TensorFact::TtTensor<float> tt_tensor = SumOfIndicesTtTensor<float>(size);

    ASSERT_TRUE(tt_tensor.NDim() == 4);

    for (std::size_t d = 0; d < 4; ++d) {
        ASSERT_TRUE(tt_tensor.Size()[d] == size[d]);
    }

    for (std::size_t d = 0; d <= 4; ++d) {
        if ((d == 0) || (d == 4)) {
            ASSERT_TRUE(tt_tensor.Rank()[d] == 1);
        } else {
            ASSERT_TRUE(tt_tensor.Rank()[d] == 2);
        }
    }

    for (std::size_t l = 0; l < 4; ++l) {
        for (std::size_t k = 0; k < 6; ++k) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t i = 0; i < 5; ++i) {
                    ASSERT_TRUE(std::abs(tt_tensor({i, j, k, l}) -
                                         (i + j + k + l)) < 1.0e-05f);
                }
            }
        }
    }
}

TEST(TtTensor, FileIO) {
    {
        const TensorFact::TtTensor<double> tt_tensor =
            SumOfIndicesTtTensor<double>({5, 3, 6, 4});
        tt_tensor.WriteToFile("tt_tensor.txt");
    }

    {
        TensorFact::TtTensor<double> tt_tensor;
        tt_tensor.ReadFromFile("tt_tensor.txt");

        for (std::size_t l = 0; l < 4; ++l) {
            for (std::size_t k = 0; k < 6; ++k) {
                for (std::size_t j = 0; j < 3; ++j) {
                    for (std::size_t i = 0; i < 5; ++i) {
                        ASSERT_TRUE(std::abs(tt_tensor({i, j, k, l}) -
                                             (i + j + k + l)) < 1.0e-15);
                    }
                }
            }
        }
    }
}

TEST(TtTensor, Addition) {
    TensorFact::TtTensor<float> tt_tensor1 =
        5.0f * SumOfIndicesTtTensor<float>({5, 3, 6, 4});
    TensorFact::TtTensor<float> tt_tensor2 =
        -2.0f * SumOfIndicesTtTensor<float>({5, 3, 6, 4});

    TensorFact::TtTensor<float> tt_tensor = tt_tensor1 + tt_tensor2;

    for (std::size_t l = 0; l < 4; ++l) {
        for (std::size_t k = 0; k < 6; ++k) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t i = 0; i < 5; ++i) {
                    ASSERT_TRUE(std::abs(tt_tensor({i, j, k, l}) -
                                         3.0f * (i + j + k + l)) < 1.0e-05f);
                }
            }
        }
    }
}

TEST(TtTensor, ScalarMultiplication) {
    TensorFact::TtTensor<double> tt_tensor1 =
        SumOfIndicesTtTensor<double>({5, 3, 6, 4});
    TensorFact::TtTensor<double> tt_tensor2 = 2.0 * tt_tensor1;

    for (std::size_t l = 0; l < 4; ++l) {
        for (std::size_t k = 0; k < 6; ++k) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t i = 0; i < 5; ++i) {
                    ASSERT_TRUE(std::abs(tt_tensor2({i, j, k, l}) -
                                         2.0 * (i + j + k + l)) < 1.0e-15);
                }
            }
        }
    }
}

TEST(TtTensor, ComputeFromFull) {
    TensorFact::Array<float> array;
    array.Resize({3, 4, 5, 6});
    for (std::size_t l = 0; l < 6; ++l) {
        for (std::size_t k = 0; k < 5; ++k) {
            for (std::size_t j = 0; j < 4; ++j) {
                for (std::size_t i = 0; i < 3; ++i) {
                    array({i, j, k, l}) = i + j + k + l;
                }
            }
        }
    }

    float rel_acc = 1.0e-06f;

    TensorFact::TtTensor<float> tt_tensor;
    tt_tensor.ComputeFromFull(array, rel_acc);

    const std::vector<std::size_t> &ranks = tt_tensor.Rank();
    ASSERT_EQ(ranks[0], 1);
    ASSERT_EQ(ranks[1], 2);
    ASSERT_EQ(ranks[2], 2);
    ASSERT_EQ(ranks[3], 2);
    ASSERT_EQ(ranks[4], 1);

    float absolute_error = 0.0f;
    for (std::size_t l = 0; l < 6; ++l) {
        for (std::size_t k = 0; k < 5; ++k) {
            for (std::size_t j = 0; j < 4; ++j) {
                for (std::size_t i = 0; i < 3; ++i) {
                    absolute_error +=
                        std::pow(i + j + k + l - tt_tensor({i, j, k, l}), 2.0f);
                }
            }
        }
    }
    absolute_error = std::sqrt(absolute_error);

    ASSERT_TRUE(absolute_error <= rel_acc * array.FrobeniusNorm());
}

TEST(TtTensor, Round) {
    const TensorFact::TtTensor<double> tt_tensor =
        SumOfIndicesTtTensor<double>({5, 3, 6, 4});

    const TensorFact::TtTensor<double> tt_tensor_1 = 2.0 * tt_tensor;

    TensorFact::TtTensor<double> tt_tensor_2 = tt_tensor + tt_tensor;
    tt_tensor_2.Round(1.0e-15);

    for (std::size_t l = 0; l < 4; ++l) {
        for (std::size_t k = 0; k < 6; ++k) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t i = 0; i < 5; ++i) {
                    ASSERT_TRUE(std::abs(tt_tensor_1({i, j, k, l}) -
                                         tt_tensor_2({i, j, k, l})) < 1.0e-13);
                }
            }
        }
    }

    const std::size_t &ndim = tt_tensor_1.NDim();

    const std::vector<std::size_t> &rank_1 = tt_tensor_1.Rank();
    const std::vector<std::size_t> &rank_2 = tt_tensor_2.Rank();

    for (std::size_t d = 0; d <= ndim; ++d) {
        ASSERT_EQ(rank_1[d], rank_2[d]);
    }
}

TEST(TtTensor, Concatenate) {
    const TensorFact::TtTensor<float> tt_tensor_1 =
        SumOfIndicesTtTensor<float>({5, 3, 6, 4});
    const TensorFact::TtTensor<float> tt_tensor_2 =
        SumOfIndicesTtTensor<float>({5, 7, 6, 4});

    const TensorFact::TtTensor<float> tt_tensor =
        tt_tensor_1.Concatenate(tt_tensor_2, 1, 1.0e-05f);

    for (std::size_t l = 0; l < 4; ++l) {
        for (std::size_t k = 0; k < 6; ++k) {
            for (std::size_t j = 0; j < 10; ++j) {
                for (std::size_t i = 0; i < 5; ++i) {
                    if (j < 3) {
                        ASSERT_TRUE(std::abs(tt_tensor({i, j, k, l}) -
                                             tt_tensor_1({i, j, k, l})) <
                                    1.0e-04f);
                    } else {
                        ASSERT_TRUE(std::abs(tt_tensor({i, j, k, l}) -
                                             tt_tensor_2({i, j - 3, k, l})) <
                                    1.0e-04f);
                    }
                }
            }
        }
    }
}

TEST(TtTensor, DotProduct) {
    TensorFact::TtTensor<double> tt_tensor1 =
        SumOfIndicesTtTensor<double>({5, 3, 6, 4});
    TensorFact::TtTensor<double> tt_tensor2 =
        -2.0 * SumOfIndicesTtTensor<double>({5, 3, 6, 4});

    double obtained_value = tt_tensor1.Dot(tt_tensor2);

    double expected_value = 0.0;
    for (std::size_t l = 0; l < 4; ++l) {
        for (std::size_t k = 0; k < 6; ++k) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t i = 0; i < 5; ++i) {
                    expected_value += std::pow(i + j + k + l, 2.0);
                }
            }
        }
    }
    expected_value *= -2.0;

    ASSERT_TRUE(std::abs(obtained_value - expected_value) < 1.0e-15);
}

TEST(TtTensor, FrobeniusNorm) {
    TensorFact::TtTensor<float> tt_tensor =
        SumOfIndicesTtTensor<float>({5, 3, 6, 4});

    float obtained_value = tt_tensor.FrobeniusNorm();

    float expected_value = 0.0f;
    for (std::size_t l = 0; l < 4; ++l) {
        for (std::size_t k = 0; k < 6; ++k) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t i = 0; i < 5; ++i) {
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
