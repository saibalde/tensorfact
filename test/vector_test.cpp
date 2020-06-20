#include "tt/vector.hpp"

#include <armadillo>
#include <iostream>

#include "gtest/gtest.h"

TEST(vector, construct_from_dims_and_ranks) {
  arma::Col<arma::uword> dims({3, 4, 5});
  arma::Col<arma::uword> ranks({1, 2, 3, 1});

  tt::Vector tt_vector(dims, ranks);

  ASSERT_TRUE(tt_vector.NumDims() == 3);
  ASSERT_TRUE(arma::all(tt_vector.Dims() == dims));
  ASSERT_TRUE(arma::all(tt_vector.Ranks() == ranks));
  ASSERT_TRUE(tt_vector.MaxRank() == 3);
}

TEST(vector, construct_from_cores) {
  arma::field<arma::Cube<double>> cores(3);

  cores(0).set_size(1, 3, 2);
  for (arma::uword i = 0; i < 3; ++i) {
    cores(0)(0, i, 0) = i;
    cores(0)(0, i, 1) = 1;
  }

  cores(1).set_size(2, 4, 2);
  for (arma::uword i = 0; i < 4; ++i) {
    cores(1)(0, i, 0) = 1;
    cores(1)(1, i, 0) = i;
    cores(1)(0, i, 1) = 0;
    cores(1)(1, i, 1) = 1;
  }

  cores(2).set_size(2, 5, 1);
  for (arma::uword i = 0; i < 5; ++i) {
    cores(2)(0, i, 0) = 1;
    cores(2)(1, i, 0) = i;
  }

  tt::Vector tt_vector(cores);

  ASSERT_TRUE(tt_vector.NumDims() == 3);
  ASSERT_TRUE(arma::all(tt_vector.Dims() == arma::Col<arma::uword>({3, 4, 5})));
  ASSERT_TRUE(
      arma::all(tt_vector.Ranks() == arma::Col<arma::uword>({1, 2, 2, 1})));
  ASSERT_TRUE(tt_vector.MaxRank() == 2);

  for (arma::uword k = 0; k < 5; ++k) {
    for (arma::uword j = 0; j < 4; ++j) {
      for (arma::uword i = 0; i < 3; ++i) {
        ASSERT_TRUE(std::abs(tt_vector({i, j, k}) - i - j - k) < 1.0e-12);
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
