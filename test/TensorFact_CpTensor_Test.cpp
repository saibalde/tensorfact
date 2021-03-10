#include "TensorFact_CpTensor.hpp"

#include <cmath>
#include <vector>

#include "TensorFact_Array.hpp"
#include "gtest/gtest.h"

template <typename Scalar>
TensorFact::CpTensor<Scalar> ProductOfIndicesCpTensor(
    const std::vector<std::size_t> size) {
    const std::size_t ndim = size.size();

    std::vector<TensorFact::Array<Scalar>> factor(ndim);

    for (std::size_t d = 0; d < ndim; ++d) {
        factor[d].Resize({size[d], 1});

        for (std::size_t i = 0; i < size[d]; ++i) {
            factor[d]({i, 0}) = static_cast<Scalar>(i);
        }
    }

    return TensorFact::CpTensor<Scalar>(factor);
}

TEST(CpTensor, ConstructFromFactor) {
    const TensorFact::CpTensor<float> cp_tensor =
        ProductOfIndicesCpTensor<float>({2, 3, 4, 5});

    for (std::size_t l = 0; l < 5; ++l) {
        for (std::size_t k = 0; k < 4; ++k) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t i = 0; i < 2; ++i) {
                    ASSERT_TRUE(std::abs(cp_tensor({i, j, k, l}) -
                                         i * j * k * l) < 1.0e-06);
                }
            }
        }
    }
}

TEST(CpTensor, FileIO) {
    {
        const TensorFact::CpTensor<double> cp_tensor =
            ProductOfIndicesCpTensor<double>({3, 2, 5, 4});
        cp_tensor.WriteToFile("cp_tensor.txt");
    }

    {
        TensorFact::CpTensor<double> cp_tensor;
        cp_tensor.ReadFromFile("cp_tensor.txt");

        for (std::size_t l = 0; l < 4; ++l) {
            for (std::size_t k = 0; k < 5; ++k) {
                for (std::size_t j = 0; j < 2; ++j) {
                    for (std::size_t i = 0; i < 3; ++i) {
                        ASSERT_TRUE(std::abs(cp_tensor({i, j, k, l}) -
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
