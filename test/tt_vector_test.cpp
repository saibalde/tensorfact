#include "tt_vector.hpp"

#include <armadillo>
#include <cmath>

#include "gtest/gtest.h"

/**
 * Create TT-Vector \f$v(i_0, ..., i_{d - 1}) = i_0 + ... + i_{d - 1}\f$ given
 * the dimensions.
 */
TtVector CreateTestTtVector(const arma::Col<arma::uword> &dims) {
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

  return TtVector(cores);
}

TEST(vector, construct) {
  TtVector tt_vector = CreateTestTtVector({5, 3, 6, 4});

  ASSERT_TRUE(tt_vector.NumDims() == 4);
  ASSERT_TRUE(
      arma::all(tt_vector.Dims() == arma::Col<arma::uword>({5, 3, 6, 4})));
  ASSERT_TRUE(
      arma::all(tt_vector.Ranks() == arma::Col<arma::uword>({1, 2, 2, 2, 1})));

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

TEST(vector, vector_addition) {
  TtVector tt_vector1 = 5.0 * CreateTestTtVector({5, 3, 6, 4});
  TtVector tt_vector2 = -2.0 * CreateTestTtVector({5, 3, 6, 4});

  TtVector tt_vector = tt_vector1 + tt_vector2;

  for (arma::uword l = 0; l < 4; ++l) {
    for (arma::uword k = 0; k < 6; ++k) {
      for (arma::uword j = 0; j < 3; ++j) {
        for (arma::uword i = 0; i < 5; ++i) {
          ASSERT_TRUE(std::abs(tt_vector({i, j, k, l}) -
                               3.0 * (i + j + k + l)) < 1.0e-15);
        }
      }
    }
  }
}

TEST(vector, scalar_multiplication) {
  TtVector tt_vector1 = CreateTestTtVector({5, 3, 6, 4});
  TtVector tt_vector2 = 2.0 * tt_vector1;

  for (arma::uword l = 0; l < 4; ++l) {
    for (arma::uword k = 0; k < 6; ++k) {
      for (arma::uword j = 0; j < 3; ++j) {
        for (arma::uword i = 0; i < 5; ++i) {
          ASSERT_TRUE(std::abs(tt_vector2({i, j, k, l}) -
                               2.0 * (i + j + k + l)) < 1.0e-15);
        }
      }
    }
  }
}

TEST(vector, dot_product) {
  TtVector tt_vector1 = CreateTestTtVector({5, 3, 6, 4});
  TtVector tt_vector2 = -2.0 * CreateTestTtVector({5, 3, 6, 4});

  double obtained_value = Dot(tt_vector1, tt_vector2);

  double expected_value = 0.0;

  for (arma::uword l = 0; l < 4; ++l) {
    for (arma::uword k = 0; k < 6; ++k) {
      for (arma::uword j = 0; j < 3; ++j) {
        for (arma::uword i = 0; i < 5; ++i) {
          expected_value += std::pow(i + j + k + l, 2.0);
        }
      }
    }
  }

  expected_value *= -2.0;

  ASSERT_TRUE(std::abs(obtained_value - expected_value) < 1.0e-15);
}

TEST(vector, vector_norm) {
  TtVector tt_vector = CreateTestTtVector({5, 3, 6, 4});

  double obtained_value = Norm(tt_vector);

  double expected_value = 0.0;

  for (arma::uword l = 0; l < 4; ++l) {
    for (arma::uword k = 0; k < 6; ++k) {
      for (arma::uword j = 0; j < 3; ++j) {
        for (arma::uword i = 0; i < 5; ++i) {
          expected_value += std::pow(i + j + k + l, 2.0);
        }
      }
    }
  }

  expected_value = std::sqrt(expected_value);

  ASSERT_TRUE(std::abs(obtained_value - expected_value) < 1.0e-15);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
