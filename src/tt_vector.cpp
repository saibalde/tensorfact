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
  arma::Mat<double> temp;

  for (unsigned int l = 0; l < ndim_; ++l) {
    arma::uword d = ndim_ - 1 - l;

    if (d == ndim_ - 1) {
      temp = cores_(d).slice(index(d));
    } else {
      temp = cores_(d).slice(index(d)) * temp;
    }
  }

  return temp(0, 0);
}

TtVector TtVector::operator*(double constant) const {
  arma::field<arma::Cube<double>> cores_new(ndim_);

  for (arma::uword d = 0; d < ndim_; ++d) {
    cores_new(d) = cores_(d);
  }

  cores_new(ndim_ - 1) *= constant;

  return TtVector(cores_new);
}

TtVector TtVector::operator+(const TtVector &other) const {
  if (!arma::all(other.Dims() == dims_)) {
    // throw exception
  }

  // new ranks

  arma::Col<arma::uword> ranks_other = other.Ranks();

  arma::Col<arma::uword> ranks_new = ranks_ + ranks_other;
  ranks_new(0) = 1;
  ranks_new(ndim_) = 1;

  // new cores

  arma::field<arma::Cube<double>> cores_new(ndim_);

  // first core

  cores_new(0).zeros(1, ranks_new(1), dims_(0));

  for (arma::uword k = 0; k < dims_(0); ++k) {
    for (arma::uword j = 0; j < ranks_(1); ++j) {
      cores_new(0)(0, j, k) = cores_(0)(0, j, k);
    }

    for (arma::uword j = 0; j < ranks_other(1); ++j) {
      cores_new(0)(0, ranks_(1) + j, k) = other.Core(0)(0, j, k);
    }
  }

  // middle cores

  for (arma::uword d = 1; d < ndim_ - 1; ++d) {
    cores_new(d).zeros(ranks_new(d), ranks_new(d + 1), dims_(d));

    for (arma::uword k = 0; k < dims_(d); ++k) {
      for (arma::uword j = 0; j < ranks_(d + 1); ++j) {
        for (arma::uword i = 0; i < ranks_(d); ++i) {
          cores_new(d)(i, j, k) = cores_(d)(i, j, k);
        }
      }

      for (arma::uword j = 0; j < ranks_other(d + 1); ++j) {
        for (arma::uword i = 0; i < ranks_other(d); ++i) {
          cores_new(d)(ranks_(d) + i, ranks_(d + 1) + j, k) =
              other.Core(d)(i, j, k);
        }
      }
    }
  }

  // last core

  cores_new(ndim_ - 1).zeros(ranks_new(ndim_ - 1), 1, dims_(ndim_ - 1));

  for (arma::uword k = 0; k < dims_(ndim_ - 1); ++k) {
    for (arma::uword i = 0; i < ranks_(ndim_ - 1); ++i) {
      cores_new(ndim_ - 1)(i, 0, k) = cores_(ndim_ - 1)(i, 0, k);
    }

    for (arma::uword i = 0; i < ranks_other(ndim_ - 1); ++i) {
      cores_new(ndim_ - 1)(ranks_(ndim_ - 1) + i, 0, k) =
          other.Core(ndim_ - 1)(i, 0, k);
    }
  }

  return TtVector(cores_new);
}

double TtVector::Dot(const TtVector &other) const {
  if (arma::any(dims_ != other.Dims())) {
    // size mismatch
  }

  arma::Col<arma::uword> ranks_other = other.Ranks();

  arma::Cube<double> temp_3d;
  arma::Mat<double> temp_2d;

  for (unsigned int l = 0; l < ndim_; ++l) {
    arma::uword d = ndim_ - 1 - l;

    temp_3d.set_size(ranks_other(d), ranks_(d), dims_(d));

    if (d == ndim_ - 1) {
      // Kronecker product of the last cores
      for (arma::uword k = 0; k < dims_(d); ++k) {
        temp_3d.slice(k) = other.Core(d).slice(k) * cores_(d).slice(k).t();
      }
    } else {
      // multiplication by Kronecker product of the cores
      for (arma::uword k = 0; k < dims_(d); ++k) {
        temp_3d.slice(k) =
            (other.Core(d).slice(k) * temp_2d) * cores_(d).slice(k).t();
      }
    }

    temp_2d = arma::sum(temp_3d, 2);
  }

  return temp_2d(0, 0);
}
