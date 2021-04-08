#include "TensorFact_CpTensor.hpp"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

TensorFact::CpTensor ProductOfIndicesCpTensor(const std::vector<int> size) {
    const int ndim = size.size();

    int num_param = 0;
    for (int d = 0; d < ndim; ++d) {
        num_param += size[d];
    }

    std::vector<double> param(num_param);

    for (int d = 0; d < ndim; ++d) {
        int offset = 0;
        for (int k = 0; k < d; ++k) {
            offset += size[k];
        }

        for (int i = 0; i < size[d]; ++i) {
            param[offset + i] = static_cast<double>(i);
        }
    }

    return TensorFact::CpTensor(ndim, size, 1, param);
}

TEST(CpTensor, ConstructFromFactor) {
    const TensorFact::CpTensor cp_tensor =
        ProductOfIndicesCpTensor({2, 3, 4, 5});

    for (int l = 0; l < 5; ++l) {
        for (int k = 0; k < 4; ++k) {
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 2; ++i) {
                    ASSERT_NEAR(cp_tensor.Entry({i, j, k, l}), i * j * k * l,
                                1.0e-06);
                }
            }
        }
    }
}

TEST(CpTensor, FileIO) {
    {
        const TensorFact::CpTensor cp_tensor =
            ProductOfIndicesCpTensor({3, 2, 5, 4});
        cp_tensor.WriteToFile("cp_tensor.txt");
    }

    {
        TensorFact::CpTensor cp_tensor;
        cp_tensor.ReadFromFile("cp_tensor.txt");

        for (int l = 0; l < 4; ++l) {
            for (int k = 0; k < 5; ++k) {
                for (int j = 0; j < 2; ++j) {
                    for (int i = 0; i < 3; ++i) {
                        ASSERT_TRUE(std::abs(cp_tensor.Entry({i, j, k, l}) -
                                             i * j * k * l) < 1.0e-15);
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
