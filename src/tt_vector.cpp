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
