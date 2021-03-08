#include "tensorfact/cp_tensor.hpp"

#include <armadillo>
#include <cmath>

#include "gtest/gtest.h"

template <typename Real>
tensorfact::CpTensor<Real> ProductOfIndicesCpTensor(
    const arma::Col<arma::uword> size) {
    const arma::uword ndim = size.n_rows;

    arma::field<arma::Mat<Real>> factor(ndim);

    for (arma::uword d = 0; d < ndim; ++d) {
        factor(d).set_size(size(d), 1);

        for (arma::uword i = 0; i < size(d); ++i) {
            factor(d)(i, 0) = static_cast<Real>(i);
        }
    }

    return tensorfact::CpTensor<Real>(factor);
}

TEST(CpTensor, ConstructFromFactor) {
    const tensorfact::CpTensor<float> cp_tensor =
        ProductOfIndicesCpTensor<float>({2, 3, 4, 5});

    for (arma::uword l = 0; l < 5; ++l) {
        for (arma::uword k = 0; k < 4; ++k) {
            for (arma::uword j = 0; j < 3; ++j) {
                for (arma::uword i = 0; i < 2; ++i) {
                    ASSERT_TRUE(std::abs(cp_tensor({i, j, k, l}) -
                                         i * j * k * l) < 1.0e-06);
                }
            }
        }
    }
}

TEST(CpTensor, FileIO) {
    {
        const tensorfact::CpTensor<double> cp_tensor =
            ProductOfIndicesCpTensor<double>({3, 2, 5, 4});
        cp_tensor.WriteToFile("cp_tensor.txt");
    }

    {
        tensorfact::CpTensor<double> cp_tensor;
        cp_tensor.ReadFromFile("cp_tensor.txt");

        for (arma::uword l = 0; l < 4; ++l) {
            for (arma::uword k = 0; k < 5; ++k) {
                for (arma::uword j = 0; j < 2; ++j) {
                    for (arma::uword i = 0; i < 3; ++i) {
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
