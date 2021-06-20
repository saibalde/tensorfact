#include "tensorfact/tt_tensor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

template <typename Real>
tensorfact::TtTensor<Real> RandomTensor(const std::vector<long> &size,
                                        long rank) {
    const long num_dim = size.size();

    std::vector<long> rank_vector(num_dim + 1, rank);
    rank_vector[0] = 1;
    rank_vector[num_dim] = 1;

    tensorfact::TtTensor<Real> tt_tensor(num_dim, size, rank_vector);

    std::random_device device;
    std::mt19937 rng(device());
    std::uniform_real_distribution<Real> dist(-1.0, 1.0);

    for (auto &p : tt_tensor.Param()) {
        p = dist(rng);
    }

    return tt_tensor;
}

template <typename Real>
tensorfact::TtTensor<Real> IndexSumTensor(const std::vector<long> &size) {
    const long num_dim = size.size();

    std::vector<long> rank(num_dim + 1, 2);
    rank[0] = 1;
    rank[num_dim] = 1;

    tensorfact::TtTensor<Real> tt_tensor(num_dim, size, rank);

    for (long d = 0; d < num_dim; ++d) {
        for (long i = 0; i < size[d]; ++i) {
            if (d == 0) {
                tt_tensor.Param(0, i, 0, d) = i;
                tt_tensor.Param(0, i, 1, d) = 1;
            } else if (d < num_dim - 1) {
                tt_tensor.Param(0, i, 0, d) = 1;
                tt_tensor.Param(1, i, 0, d) = i;
                tt_tensor.Param(0, i, 1, d) = 0;
                tt_tensor.Param(1, i, 1, d) = 1;
            } else {
                tt_tensor.Param(0, i, 0, d) = 1;
                tt_tensor.Param(1, i, 0, d) = i;
            }
        }
    }

    return tt_tensor;
}

TEST(TtTensor, ConstructDefault) {
    tensorfact::TtTensor<float> tt_tensor_float;
    tensorfact::TtTensor<double> tt_tensor_double;
}

TEST(TtTensor, ConstructFromScalarSizeRank) {
    long num_dim = 4;
    long size = 3;
    long rank = 4;

    tensorfact::TtTensor<float> tt_tensor(num_dim, size, rank);

    ASSERT_EQ(tt_tensor.NumDim(), num_dim);

    for (long d = 0; d < num_dim; ++d) {
        ASSERT_EQ(tt_tensor.Size(d), size);
    }

    for (long d = 0; d <= num_dim; ++d) {
        if (d == 0 || d == num_dim) {
            ASSERT_EQ(tt_tensor.Rank(d), 1);
        } else {
            ASSERT_EQ(tt_tensor.Rank(d), rank);
        }
    }

    for (auto &p : tt_tensor.Param()) {
        p = 0.0f;
    }

    for (long d = 0; d < num_dim; ++d) {
        for (long k = 0; k < ((d == num_dim - 1) ? 1 : rank); ++k) {
            for (long j = 0; j < size; ++j) {
                for (long i = 0; i < ((d == 0) ? 1 : rank); ++i) {
                    ASSERT_NEAR(tt_tensor.Param(i, j, k, d), 0.0f, 1.0e-05f);
                }
            }
        }
    }

    long num_param = size * (2 * rank + (num_dim - 2) * rank * rank);
    ASSERT_EQ(tt_tensor.NumParam(), num_param);

    long num_element = std::pow(size, 4);
    ASSERT_EQ(tt_tensor.NumElement(), num_element);
}

TEST(TtTensor, ConstructFromVectorSizeRank) {
    long num_dim = 4;
    std::vector<long> size{3, 6, 8, 5};
    std::vector<long> rank{1, 4, 6, 5, 1};

    tensorfact::TtTensor<double> tt_tensor(num_dim, size, rank);

    ASSERT_EQ(tt_tensor.NumDim(), num_dim);

    for (long d = 0; d < num_dim; ++d) {
        ASSERT_EQ(tt_tensor.Size()[d], size[d]);
    }

    for (long d = 0; d <= num_dim; ++d) {
        ASSERT_EQ(tt_tensor.Rank()[d], rank[d]);
    }

    for (long d = 0; d < num_dim; ++d) {
        for (long k = 0; k < rank[d + 1]; ++k) {
            for (long j = 0; j < size[d]; ++j) {
                for (long i = 0; i < rank[d]; ++i) {
                    tt_tensor.Param(i, j, k, d) = 1.0;
                }
            }
        }
    }

    for (const auto &p : tt_tensor.Param()) {
        ASSERT_NEAR(p, 1.0, 1.0e-15);
    }
}

TEST(TtTensor, ConstructFromSizeRankParam) {
    long num_dim = 4;
    std::vector<long> size{5, 3, 6, 4};
    std::vector<long> rank{1, 6, 2, 3, 1};

    long num_param = 0;
    for (long d = 0; d < num_dim; ++d) {
        num_param += rank[d] * size[d] * rank[d + 1];
    }

    std::vector<float> param(num_param, 2.0f);

    const tensorfact::TtTensor<float> tt_tensor(num_dim, size, rank, param);

    for (const auto &p : tt_tensor.Param()) {
        ASSERT_NEAR(p, 2.0f, 1.0e-05f);
    }
}

TEST(TtTensor, ConstructCopy) {
    long num_dim = 4;
    std::vector<long> size{5, 3, 6, 4};
    std::vector<long> rank{1, 6, 2, 3, 1};

    long num_param = 0;
    for (long d = 0; d < num_dim; ++d) {
        num_param += rank[d] * size[d] * rank[d + 1];
    }

    std::vector<float> param(num_param, 2.0f);

    tensorfact::TtTensor<float> tt_tensor1(num_dim, size, rank, param);
    tensorfact::TtTensor<float> tt_tensor2 = tt_tensor1;

    for (auto &p : tt_tensor2.Param()) {
        p = 1.0f;
    }

    for (auto &p : tt_tensor1.Param()) {
        ASSERT_NEAR(p, 1.0f, 1.0e-05f);
    }
}

TEST(TtTensor, ConstructMove) {
    long num_dim = 4;
    std::vector<long> size{5, 3, 6, 4};
    std::vector<long> rank{1, 6, 2, 3, 1};

    long num_param = 0;
    for (long d = 0; d < num_dim; ++d) {
        num_param += rank[d] * size[d] * rank[d + 1];
    }

    std::vector<float> param(num_param, 2.0f);

    tensorfact::TtTensor<float> tt_tensor1(num_dim, size, rank, param);
    tensorfact::TtTensor<float> tt_tensor2 = std::move(tt_tensor1);

    for (auto &p : tt_tensor2.Param()) {
        ASSERT_NEAR(p, 2.0f, 1.0e-05f);
    }
}

TEST(TtTensor, AssignCopy) {
    long num_dim = 4;
    std::vector<long> size{5, 3, 6, 4};
    std::vector<long> rank{1, 6, 2, 3, 1};

    long num_param = 0;
    for (long d = 0; d < num_dim; ++d) {
        num_param += rank[d] * size[d] * rank[d + 1];
    }

    std::vector<float> param(num_param, 2.0f);

    tensorfact::TtTensor<float> tt_tensor1(num_dim, size, rank, param);
    tensorfact::TtTensor<float> tt_tensor2;
    tt_tensor2 = tt_tensor1;

    for (auto &p : tt_tensor2.Param()) {
        p = 1.0f;
    }

    for (auto &p : tt_tensor1.Param()) {
        ASSERT_NEAR(p, 1.0f, 1.0e-05f);
    }
}

TEST(TtTensor, AssignMove) {
    long num_dim = 4;
    std::vector<long> size{5, 3, 6, 4};
    std::vector<long> rank{1, 6, 2, 3, 1};

    long num_param = 0;
    for (long d = 0; d < num_dim; ++d) {
        num_param += rank[d] * size[d] * rank[d + 1];
    }

    std::vector<float> param(num_param, 2.0f);

    tensorfact::TtTensor<float> tt_tensor1(num_dim, size, rank, param);
    tensorfact::TtTensor<float> tt_tensor2;
    tt_tensor2 = std::move(tt_tensor1);

    for (auto &p : tt_tensor2.Param()) {
        ASSERT_NEAR(p, 2.0f, 1.0e-05f);
    }
}

TEST(TtTensor, Copy) {
    long num_dim = 4;
    std::vector<long> size{5, 3, 6, 4};
    std::vector<long> rank{1, 6, 2, 3, 1};

    long num_param = 0;
    for (long d = 0; d < num_dim; ++d) {
        num_param += rank[d] * size[d] * rank[d + 1];
    }

    std::vector<float> param(num_param, 2.0f);

    tensorfact::TtTensor<float> tt_tensor1(num_dim, size, rank, param);
    tensorfact::TtTensor<float> tt_tensor2 = tt_tensor1.Copy();

    for (auto &p : tt_tensor2.Param()) {
        p = 1.0f;
    }

    for (auto &p : tt_tensor1.Param()) {
        ASSERT_NEAR(p, 2.0f, 1.0e-05f);
    }
}

TEST(TtTensor, Addition) {
    std::vector<long> size{5, 3, 6, 4};

    tensorfact::TtTensor<float> tt_tensor1 = RandomTensor<float>(size, 2);
    tensorfact::TtTensor<float> tt_tensor2 = RandomTensor<float>(size, 3);
    tensorfact::TtTensor<float> tt_tensor3 = tt_tensor1 + tt_tensor2;

    for (long l = 0; l < size[3]; ++l) {
        for (long k = 0; k < size[2]; ++k) {
            for (long j = 0; j < size[1]; ++j) {
                for (long i = 0; i < size[0]; ++i) {
                    ASSERT_NEAR(tt_tensor1.Entry({i, j, k, l}) +
                                    tt_tensor2.Entry({i, j, k, l}),
                                tt_tensor3.Entry({i, j, k, l}), 1.0e-05f);
                }
            }
        }
    }
}

TEST(TtTensor, ScalarMultiplication) {
    std::vector<long> size{5, 3, 6, 4};

    tensorfact::TtTensor<double> tt_tensor1 = RandomTensor<double>(size, 4);
    tensorfact::TtTensor<double> tt_tensor2 = 2.0 * tt_tensor1;

    for (long l = 0; l < size[3]; ++l) {
        for (long k = 0; k < size[2]; ++k) {
            for (long j = 0; j < size[1]; ++j) {
                for (long i = 0; i < size[0]; ++i) {
                    ASSERT_NEAR(2.0 * tt_tensor1.Entry({i, j, k, l}),
                                tt_tensor2.Entry({i, j, k, l}), 1.0e-15);
                }
            }
        }
    }
}

TEST(TtTensor, ElementwiseMultiplication) {
    std::vector<long> size{4, 3, 6, 5};

    tensorfact::TtTensor<double> tt_tensor1 = RandomTensor<double>(size, 5);
    tensorfact::TtTensor<double> tt_tensor2 = RandomTensor<double>(size, 2);
    tensorfact::TtTensor<double> tt_tensor3 = tt_tensor1 * tt_tensor2;

    for (long l = 0; l < size[3]; ++l) {
        for (long k = 0; k < size[2]; ++k) {
            for (long j = 0; j < size[1]; ++j) {
                for (long i = 0; i < size[0]; ++i) {
                    ASSERT_NEAR(tt_tensor1.Entry({i, j, k, l}) *
                                    tt_tensor2.Entry({i, j, k, l}),
                                tt_tensor3.Entry({i, j, k, l}), 1.0e-14);
                }
            }
        }
    }
}

TEST(TtTensor, Concatenate) {
    std::vector<long> size1{5, 3, 6, 4};
    std::vector<long> size2{5, 7, 6, 4};

    tensorfact::TtTensor<double> tt_tensor_1 = RandomTensor<double>(size1, 3);
    tensorfact::TtTensor<double> tt_tensor_2 = RandomTensor<double>(size2, 5);

    tensorfact::TtTensor<double> tt_tensor =
        tt_tensor_1.Concatenate(tt_tensor_2, 1);
    tt_tensor.Round(1.0e-15);

    for (long l = 0; l < size1[3]; ++l) {
        for (long k = 0; k < size1[2]; ++k) {
            for (long j = 0; j < size1[1] + size2[1]; ++j) {
                for (long i = 0; i < size1[0]; ++i) {
                    if (j < size1[1]) {
                        ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}),
                                    tt_tensor_1.Entry({i, j, k, l}), 1.0e-13);
                    } else {
                        ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}),
                                    tt_tensor_2.Entry({i, j - size1[1], k, l}),
                                    1.0e-13);
                    }
                }
            }
        }
    }
}

TEST(TtTensor, Shift) {
    std::vector<long> size({3, 4, 5, 6});

    tensorfact::TtTensor<float> tt_tensor1 = RandomTensor<float>(size, 4);
    tensorfact::TtTensor<float> tt_tensor2 = tt_tensor1.Shift(1, 2);
    tensorfact::TtTensor<float> tt_tensor3 = tt_tensor1.Shift(2, -3);

    for (long l = 0; l < size[3]; ++l) {
        for (long k = 0; k < size[2]; ++k) {
            for (long j = 0; j < size[1]; ++j) {
                for (long i = 0; i < size[0]; ++i) {
                    if (j < size[1] - 2) {
                        ASSERT_NEAR(tt_tensor2.Entry({i, j, k, l}),
                                    tt_tensor1.Entry({i, j + 2, k, l}),
                                    1.0e-15);
                    } else {
                        ASSERT_NEAR(tt_tensor2.Entry({i, j, k, l}), 0.0,
                                    1.0e-15);
                    }

                    if (k >= 3) {
                        ASSERT_NEAR(tt_tensor3.Entry({i, j, k, l}),
                                    tt_tensor1.Entry({i, j, k - 3, l}),
                                    1.0e-15);
                    } else {
                        ASSERT_NEAR(tt_tensor3.Entry({i, j, k, l}), 0.0,
                                    1.0e-15);
                    }
                }
            }
        }
    }
}

TEST(TtTensor, Contract) {
    std::vector<long> size{3, 5, 6, 4};

    tensorfact::TtTensor<double> tt_tensor = IndexSumTensor<double>(size);

    double expected_value = 0.0;
    for (long l = 0; l < size[3]; ++l) {
        for (long k = 0; k < size[2]; ++k) {
            for (long j = 0; j < size[1]; ++j) {
                for (long i = 0; i < size[0]; ++i) {
                    expected_value += tt_tensor.Entry({i, j, k, l});
                }
            }
        }
    }

    std::vector<std::vector<double>> vectors(4);
    for (long d = 0; d < 4; ++d) {
        vectors[d] = std::vector<double>(size[d], 1.0);
    }

    double computed_value = tt_tensor.Contract(vectors);

    EXPECT_NEAR(expected_value, computed_value, 1.0e-15);
}

TEST(TtTensor, Dot) {
    std::vector<long> size{5, 3, 6, 4};

    tensorfact::TtTensor<float> tt_tensor1 = IndexSumTensor<float>(size);
    tensorfact::TtTensor<float> tt_tensor2 = -2.0f * tt_tensor1;

    float obtained_value = tt_tensor1.Dot(tt_tensor2);

    float expected_value = 0.0f;
    for (long l = 0; l < size[3]; ++l) {
        for (long k = 0; k < size[2]; ++k) {
            for (long j = 0; j < size[1]; ++j) {
                for (long i = 0; i < size[0]; ++i) {
                    expected_value += std::pow(i + j + k + l, 2.0f);
                }
            }
        }
    }
    expected_value *= -2.0f;

    ASSERT_NEAR(obtained_value, expected_value, 1.0e-05f);
}

TEST(TtTensor, FrobeniusNorm) {
    std::vector<long> size{5, 3, 6, 4};

    tensorfact::TtTensor<double> tt_tensor = IndexSumTensor<double>(size);

    double obtained_value = tt_tensor.FrobeniusNorm();

    double expected_value = 0.0;
    for (long l = 0; l < size[3]; ++l) {
        for (long k = 0; k < size[2]; ++k) {
            for (long j = 0; j < size[1]; ++j) {
                for (long i = 0; i < size[0]; ++i) {
                    expected_value += std::pow(i + j + k + l, 2.0);
                }
            }
        }
    }
    expected_value = std::sqrt(expected_value);

    ASSERT_TRUE(std::abs(obtained_value - expected_value) < 1.0e-15);
}

TEST(TtTensor, Round) {
    std::vector<long> size{5, 3, 6, 4};

    tensorfact::TtTensor<float> tt_tensor = RandomTensor<float>(size, 4);

    tensorfact::TtTensor<float> tt_tensor_1 = 2.0f * tt_tensor;
    tensorfact::TtTensor<float> tt_tensor_2 = tt_tensor + tt_tensor;

    tt_tensor_2.Round(1.0e-05f);

    for (long l = 0; l < size[3]; ++l) {
        for (long k = 0; k < size[2]; ++k) {
            for (long j = 0; j < size[1]; ++j) {
                for (long i = 0; i < size[0]; ++i) {
                    ASSERT_NEAR(tt_tensor_1.Entry({i, j, k, l}),
                                tt_tensor_2.Entry({i, j, k, l}), 1.0e-04f);
                }
            }
        }
    }

    const auto &rank_1 = tt_tensor_1.Rank();
    const auto &rank_2 = tt_tensor_2.Rank();
    for (long d = 0; d <= 4; ++d) {
        ASSERT_EQ(rank_1[d], rank_2[d]);
    }
}

TEST(TtTensor, Full) {
    std::vector<long> size{3, 5, 4, 6};

    tensorfact::TtTensor<float> tt_tensor = IndexSumTensor<float>(size);
    std::vector<float> tensor = tt_tensor.Full();

    for (long l = 0; l < size[3]; ++l) {
        for (long k = 0; k < size[2]; ++k) {
            for (long j = 0; j < size[1]; ++j) {
                for (long i = 0; i < size[0]; ++i) {
                    ASSERT_NEAR(tensor[i + size[0] * j + size[0] * size[1] * k +
                                       size[0] * size[1] * size[2] * l],
                                i + j + k + l, 1.0e-05);
                }
            }
        }
    }
}

TEST(TtTensor, Entry) {
    std::vector<long> size{6, 4, 5, 3};

    tensorfact::TtTensor<float> tt_tensor = IndexSumTensor<float>(size);

    for (long l = 0; l < size[3]; ++l) {
        for (long k = 0; k < size[2]; ++k) {
            for (long j = 0; j < size[1]; ++j) {
                for (long i = 0; i < size[0]; ++i) {
                    ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}), i + j + k + l,
                                1.0e-05f);
                }
            }
        }
    }
}

TEST(TtTensor, TextIO) {
    std::vector<long> size{4, 3, 6, 5};

    {
        tensorfact::TtTensor<double> tt_tensor = IndexSumTensor<double>(size);
        tt_tensor.WriteToFile("tt_tensor.txt");
    }

    {
        tensorfact::TtTensor<double> tt_tensor;
        tt_tensor.ReadFromFile("tt_tensor.txt");

        for (long l = 0; l < size[3]; ++l) {
            for (long k = 0; k < size[2]; ++k) {
                for (long j = 0; j < size[1]; ++j) {
                    for (long i = 0; i < size[0]; ++i) {
                        ASSERT_NEAR(tt_tensor.Entry({i, j, k, l}),
                                    i + j + k + l, 1.0e-15);
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
