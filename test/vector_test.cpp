#include "tt/vector.hpp"

#include <armadillo>
#include <iostream>

#include "gtest/gtest.h"

TEST(vector, construct_from_dims_and_ranks) {
  arma::Col<arma::uword> dims({3, 7, 5});
  arma::Col<arma::uword> ranks({1, 2, 3, 1});

  tt::Vector tt_vector(dims, ranks);

  ASSERT_TRUE(tt_vector.NumDims() == 3);
  ASSERT_TRUE(arma::all(tt_vector.Dims() == dims));
  ASSERT_TRUE(arma::all(tt_vector.Ranks() == ranks));
  ASSERT_TRUE(tt_vector.MaxRank() == 3);

  ASSERT_TRUE(arma::all(arma::vectorise(
      arma::abs(tt_vector.Core(0) -
                arma::Cube<double>(1, 2, 3, arma::fill::zeros)) < 1.0e-15)));
  ASSERT_TRUE(arma::all(arma::vectorise(
      arma::abs(tt_vector.Core(1) -
                arma::Cube<double>(2, 3, 7, arma::fill::zeros)) < 1.0e-15)));
  ASSERT_TRUE(arma::all(arma::vectorise(
      arma::abs(tt_vector.Core(2) -
                arma::Cube<double>(3, 1, 5, arma::fill::zeros)) < 1.0e-15)));
}

TEST(vector, construct_from_cores) {
  arma::Col<arma::uword> dims{5, 3, 6, 4};

  arma::uword ndim = dims.n_elem;

  arma::Col<arma::uword> ranks(ndim + 1);
  ranks(0) = 1;
  for (arma::uword d = 1; d < ndim; ++d) {
    ranks(d) = 2;
  }
  ranks(ndim) = 1;

  arma::field<arma::Cube<double>> cores(ndim);

  cores(0).zeros(1, 2, dims(0));
  for (arma::uword i = 0; i < dims(0); ++i) {
    cores(0)(0, 0, i) = i;
    cores(0)(0, 1, i) = 1;
  }

  for (arma::uword d = 1; d < ndim - 1; ++d) {
    cores(d).zeros(2, 2, dims(d));
    for (arma::uword i = 0; i < dims(d); ++i) {
      cores(d)(0, 0, i) = 1;
      cores(d)(1, 0, i) = i;
      cores(d)(0, 1, i) = 0;
      cores(d)(1, 1, i) = 1;
    }
  }

  cores(ndim - 1).set_size(2, 1, dims(ndim - 1));
  for (arma::uword i = 0; i < dims(ndim - 1); ++i) {
    cores(ndim - 1)(0, 0, i) = 1;
    cores(ndim - 1)(1, 0, i) = i;
  }

  tt::Vector tt_vector(cores);

  ASSERT_TRUE(tt_vector.NumDims() == ndim);
  ASSERT_TRUE(arma::all(tt_vector.Dims() == dims));
  ASSERT_TRUE(arma::all(tt_vector.Ranks() == ranks));
  ASSERT_TRUE(tt_vector.MaxRank() == 2);

  for (arma::uword l = 0; l < 4; ++l) {
    for (arma::uword k = 0; k < 6; ++k) {
      for (arma::uword j = 0; j < 3; ++j) {
        for (arma::uword i = 0; i < 5; ++i) {
          ASSERT_TRUE(std::abs(tt_vector({i, j, k, l}) - i - j - k - l) <
                      1.0e-15);
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
