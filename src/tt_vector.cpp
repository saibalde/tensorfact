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

  for (arma::uword i = 0; i < ndim_; ++i) {
    dims_[i] = cores[i].n_slices;

    ranks_[i] = cores[i].n_rows;
    if ((i > 0) && (cores[i - 1].n_cols != ranks_[i])) {
      // raise error
      // core dimensionality is not compatible with TT format
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

TtVector operator+(const TtVector &vector1, const TtVector &vector2) {
  if (!arma::all(vector1.dims_ == vector2.dims_)) {
    // throw exception
  }

  // new ranks

  arma::Col<arma::uword> ranks_new = vector1.ranks_ + vector2.ranks_;
  ranks_new(0) = 1;
  ranks_new(vector1.ndim_) = 1;

  // new cores

  arma::field<arma::Cube<double>> cores_new(vector1.ndim_);

  // first core

  cores_new(0).zeros(1, ranks_new(1), vector1.dims_(0));

  for (arma::uword k = 0; k < vector1.dims_(0); ++k) {
    for (arma::uword j = 0; j < vector1.ranks_(1); ++j) {
      cores_new(0)(0, j, k) = vector1.cores_(0)(0, j, k);
    }

    for (arma::uword j = 0; j < vector2.ranks_(1); ++j) {
      cores_new(0)(0, vector1.ranks_(1) + j, k) = vector2.cores_(0)(0, j, k);
    }
  }

  // middle cores

  for (arma::uword d = 1; d < vector1.ndim_ - 1; ++d) {
    cores_new(d).zeros(ranks_new(d), ranks_new(d + 1), vector1.dims_(d));

    for (arma::uword k = 0; k < vector1.dims_(d); ++k) {
      for (arma::uword j = 0; j < vector1.ranks_(d + 1); ++j) {
        for (arma::uword i = 0; i < vector1.ranks_(d); ++i) {
          cores_new(d)(i, j, k) = vector1.cores_(d)(i, j, k);
        }
      }

      for (arma::uword j = 0; j < vector2.ranks_(d + 1); ++j) {
        for (arma::uword i = 0; i < vector2.ranks_(d); ++i) {
          cores_new(d)(vector1.ranks_(d) + i, vector1.ranks_(d + 1) + j, k) =
              vector2.cores_(d)(i, j, k);
        }
      }
    }
  }

  // last core

  cores_new(vector1.ndim_ - 1)
      .zeros(ranks_new(vector1.ndim_ - 1), 1, vector1.dims_(vector1.ndim_ - 1));

  for (arma::uword k = 0; k < vector1.dims_(vector1.ndim_ - 1); ++k) {
    for (arma::uword i = 0; i < vector1.ranks_(vector1.ndim_ - 1); ++i) {
      cores_new(vector1.ndim_ - 1)(i, 0, k) =
          vector1.cores_(vector1.ndim_ - 1)(i, 0, k);
    }

    for (arma::uword i = 0; i < vector2.ranks_(vector1.ndim_ - 1); ++i) {
      cores_new(vector1.ndim_ - 1)(vector1.ranks_(vector1.ndim_ - 1) + i, 0,
                                   k) =
          vector2.cores_(vector1.ndim_ - 1)(i, 0, k);
    }
  }

  return TtVector(cores_new);
}

TtVector operator*(double constant, const TtVector &vector) {
  arma::field<arma::Cube<double>> cores_new(vector.ndim_);

  for (arma::uword d = 0; d < vector.ndim_; ++d) {
    cores_new(d) = vector.cores_(d);
  }

  cores_new(vector.ndim_ - 1) *= constant;

  return TtVector(cores_new);
}

double Dot(const TtVector &vector1, const TtVector &vector2) {
  if (arma::any(vector1.dims_ != vector2.dims_)) {
    // size mismatch
  }

  arma::Cube<double> temp_3d;
  arma::Mat<double> temp_2d;

  for (unsigned int l = 0; l < vector1.ndim_; ++l) {
    arma::uword d = vector1.ndim_ - 1 - l;

    temp_3d.set_size(vector2.ranks_(d), vector1.ranks_(d), vector1.dims_(d));

    if (d == vector1.ndim_ - 1) {
      // Kronecker product of the last cores
      for (arma::uword k = 0; k < vector1.dims_(d); ++k) {
        temp_3d.slice(k) =
            vector2.cores_(d).slice(k) * vector1.cores_(d).slice(k).t();
      }
    } else {
      // multiplication by Kronecker product of the cores
      for (arma::uword k = 0; k < vector1.dims_(d); ++k) {
        temp_3d.slice(k) = vector2.cores_(d).slice(k) *
                           (temp_2d * vector1.cores_(d).slice(k).t());
      }
    }

    temp_2d = arma::sum(temp_3d, 2);
  }

  return temp_2d(0, 0);
}
