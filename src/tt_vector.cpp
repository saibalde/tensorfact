#include "tt_vector.hpp"

#include <stdexcept>

TtVector::TtVector(const arma::Col<arma::uword> &dims,
                   const arma::Col<arma::uword> &ranks) {
  ndim_ = dims.n_elem;

  dims_ = dims;

  if ((ranks.n_elem != ndim_ + 1) || (ranks[0] != 1) || (ranks[ndim_] != 1)) {
    // raise error
    // dimension and rank specifications are incompatible with TT format
  }
  ranks_ = ranks;

  maxrank_ = 1;
  for (arma::uword i = 1; i < ndim_; ++i) {
    if (ranks[i] > maxrank_) {
      maxrank_ = ranks[i];
    }
  }

  cores_.set_size(ndim_);
  for (arma::uword i = 0; i < ndim_; ++i) {
    cores_[i].zeros(ranks[i], ranks[i + 1], dims[i]);
  }
}

TtVector::TtVector(const arma::field<arma::Cube<double>> &cores)
    : cores_(cores) {
  ndim_ = cores.n_elem;

  if ((cores[0].n_rows != 1) || (cores[ndim_ - 1].n_cols != 1)) {
    // raise error
    // core dimensionality is not compatible with TT format
  }

  dims_.zeros(ndim_);
  ranks_.zeros(ndim_ + 1);
  maxrank_ = 1;

  for (arma::uword i = 0; i < ndim_; ++i) {
    dims_[i] = cores[i].n_slices;

    ranks_[i] = cores[i].n_rows;
    if ((i > 0) && (cores[i - 1].n_cols != ranks_[i])) {
      // raise error
      // core dimensionality is not compatible with TT format
    }

    if (ranks_[i] > maxrank_) {
      maxrank_ = ranks_[i];
    }
  }
  ranks_[ndim_] = 1;

  cores_ = cores;
}

double TtVector::operator()(const arma::Col<arma::uword> &index) const {
  arma::Col<double> entries(2 * maxrank_);
  unsigned int skip = 0;

  for (unsigned int l = 0; l < ndim_; ++l) {
    arma::uword d = ndim_ - 1 - l;

    unsigned int next_skip = (skip + 1) % 2;

    if (d == ndim_ - 1) {
      // copy over
      for (arma::uword i = 0; i < ranks_(d); ++i) {
        entries(next_skip * maxrank_ + i) = cores_(d)(i, 0, index[d]);
      }
    } else {
      // multiplication by core
      for (arma::uword i = 0; i < ranks_(d); ++i) {
        entries(next_skip * maxrank_ + i) = 0.0;
        for (arma::uword j = 0; j < ranks_(d + 1); ++j) {
          entries(next_skip * maxrank_ + i) +=
              cores_(d)(i, j, index(d)) * entries(skip * maxrank_ + j);
        }
      }
    }

    skip = next_skip;
  }

  return entries(skip * maxrank_);
}

TtVector operator*(double constant, const TtVector &vector) {
  arma::uword ndim = vector.NumDims();
  arma::field<arma::Cube<double>> cores(ndim);

  for (arma::uword d = 0; d < ndim; ++d) {
    cores(d) = vector.Core(d);
  }

  cores(ndim - 1) *= constant;

  return TtVector(cores);
}

TtVector operator+(const TtVector &vector1, const TtVector &vector2) {
  arma::Col<arma::uword> dims = vector1.Dims();

  if (!arma::all(dims == vector2.Dims())) {
    // throw exception
  }

  arma::uword ndim = vector1.NumDims();

  arma::field<arma::Cube<double>> cores(ndim);

  arma::Col<arma::uword> ranks1 = vector1.Ranks();
  arma::Col<arma::uword> ranks2 = vector2.Ranks();

  arma::Col<arma::uword> ranks = ranks1 + ranks2;
  ranks(0) = 1;
  ranks(ndim) = 1;

  // first core

  cores(0).zeros(1, ranks(1), dims(0));

  for (arma::uword k = 0; k < dims(0); ++k) {
    for (arma::uword j = 0; j < ranks1(1); ++j) {
      cores(0)(0, j, k) = vector1.Core(0)(0, j, k);
    }

    for (arma::uword j = 0; j < ranks2(1); ++j) {
      cores(0)(0, ranks1(1) + j, k) = vector2.Core(0)(0, j, k);
    }
  }

  // middle cores

  for (arma::uword d = 1; d < ndim - 1; ++d) {
    cores(d).zeros(ranks(d), ranks(d + 1), dims(d));

    for (arma::uword k = 0; k < dims(d); ++k) {
      for (arma::uword j = 0; j < ranks1(d + 1); ++j) {
        for (arma::uword i = 0; i < ranks1(d); ++i) {
          cores(d)(i, j, k) = vector1.Core(d)(i, j, k);
        }
      }

      for (arma::uword j = 0; j < ranks2(d + 1); ++j) {
        for (arma::uword i = 0; i < ranks2(d); ++i) {
          cores(d)(ranks1(d) + i, ranks1(d + 1) + j, k) =
              vector2.Core(d)(i, j, k);
        }
      }
    }
  }

  // last core

  cores(ndim - 1).zeros(ranks(ndim - 1), 1, dims(ndim - 1));

  for (arma::uword k = 0; k < dims(ndim - 1); ++k) {
    for (arma::uword i = 0; i < ranks1(ndim - 1); ++i) {
      cores(ndim - 1)(i, 0, k) = vector1.Core(ndim - 1)(i, 0, k);
    }

    for (arma::uword i = 0; i < ranks2(ndim - 1); ++i) {
      cores(ndim - 1)(ranks1(ndim - 1) + i, 0, k) =
          vector2.Core(ndim - 1)(i, 0, k);
    }
  }

  return TtVector(cores);
}
