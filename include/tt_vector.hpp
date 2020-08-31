#ifndef TT_VECTOR_HPP
#define TT_VECTOR_HPP

#include <armadillo>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "truncated_svd.hpp"

/// @brief TT representation of a multidimensional vector
///
/// A TT-vector is a memory-efficient representation of a multidimensional array
/// \f$v(i_0, \ldots, i_{d - 1})\f$ where each of the entries are computed as
/// \f[
///     v(i_0, \ldots, i_{d - 1}) = v_0(i_0) \cdots v_{d - 1}(i_{d - 1})
/// \f]
/// Here \f$v_k(i_k)\f$ is the \f$i_k\f$-th slice of the 3D array \f$v_k\f$,
/// also referred to as the \f$k\f$-th TT core of \f$v\f$. Each of these slices
/// are \f$r_k \times r_{k + 1}\f$ dimensional matrices, with \f$r_0 = r_d =
/// 1\f$. Assuming \f$n_k \sim n\f$ and \f$r_k \sim r\f$, this reduces the
/// storage complexity \f$\mathcal{O}(n^d)\f$ of the full tensor to
/// \f$\mathcal{O}(d n r^2)\f$ in the TT format.
template <typename Real, typename Index>
class TtVector {
public:
    /// Default constructor
    TtVector() = default;

    /// Construct a TT-vector from the cores
    TtVector(const arma::field<arma::Cube<Real>> &cores);

    /// Construct a TT-vector from full tensor using TT-SVD
    TtVector(const arma::Col<Real> &array, const arma::Col<Index> &size,
             Real relAcc = 10 * std::numeric_limits<Real>::epsilon());

    /// Default destructor
    ~TtVector() = default;

    /// @name Attribute Access
    ///@{

    /// Return the dimensionality of the TT-vector
    const Index &ndim() const { return ndim_; }

    /// Return the size of the TT-vector
    const arma::Col<Index> &size() const { return size_; }

    /// Return the TT-ranks of the TT-vector
    const arma::Col<Index> &ranks() const { return ranks_; }

    /// Return the specified core of the TT-vector
    const arma::Cube<Real> &core(Index i) const { return cores_(i); }

    /// Compute and return the entry of the TT-vector at given index
    Real operator()(const arma::Col<Index> &index) const;

    ///@}

    /// @name Arithmatic Operations
    ///@{

    /// TT-vector addition
    TtVector<Real, Index> operator+(const TtVector<Real, Index> &other) const;

    /// TT-vector subtraction
    TtVector<Real, Index> operator-(const TtVector<Real, Index> &other) const;

    /// TT-vector scalar multiplication
    TtVector<Real, Index> operator*(Real constant) const;

    /// TT-vector scalar division
    TtVector<Real, Index> operator/(Real constant) const;

    ///@}

    /// @name Mathematical Functions
    ///@{

    /// TT-vector dot product
    Real dot(const TtVector<Real, Index> &other) const;

    /// TT-vector 2-norm
    Real norm2() const;

    ///@}

private:
    Index ndim_;
    arma::Col<Index> size_;
    arma::Col<Index> ranks_;
    arma::field<arma::Cube<Real>> cores_;
};

template <typename Real, typename Index>
TtVector<Real, Index>::TtVector(const arma::field<arma::Cube<Real>> &cores)
    : cores_(cores) {
    ndim_ = cores.n_elem;

    if ((cores[0].n_rows != 1) || (cores[ndim_ - 1].n_cols != 1)) {
        throw std::logic_error(
            "TtVector::TtVector() - Dimensionality of the cores is not "
            "compatible with TT format");
    }

    size_.zeros(ndim_);
    ranks_.zeros(ndim_ + 1);

    for (Index i = 0; i < ndim_; ++i) {
        size_[i] = cores[i].n_slices;

        ranks_[i] = cores[i].n_rows;
        if ((i > 0) && (cores[i - 1].n_cols != ranks_[i])) {
            throw std::logic_error(
                "TtVector::TtVector() - Dimensionality of the cores is not "
                "compatible with TT format");
        }
    }
    ranks_[ndim_] = 1;

    cores_ = cores;
}

template <typename Real, typename Index>
TtVector<Real, Index>::TtVector(const arma::Col<Real> &array,
                                const arma::Col<Index> &size, Real relAcc) {
    if (!arma::all(size > 0)) {
        throw std::logic_error(
            "TtVector::TtVector() - Entries of the size vector must be "
            "positive");
    }

    if (array.n_elem != arma::prod(size)) {
        throw std::logic_error(
            "TtVector::TtVector() - Number of array elements and array size "
            "does not match");
    }

    if (relAcc < std::numeric_limits<Real>::epsilon()) {
        throw std::logic_error(
            "TtVector::TtVector() - Relative accuracy is too small");
    }

    ndim_ = size.n_elem;
    size_ = size;
    ranks_ = arma::Col<Index>(ndim_ + 1);
    cores_ = arma::field<arma::Cube<Real>>(ndim_);

    const Real deltaSquare =
        std::pow(relAcc, 2) * arma::dot(array, array) / (ndim_ - 1);

    arma::Col<Real> arrayCopy(array);
    ranks_(0) = 1;
    for (Index d = 0; d < ndim_; ++d) {
        if (d < ndim_ - 1) {
            arma::Mat<Real> C(arrayCopy.memptr(), ranks_(d) * size_(d),
                              arrayCopy.n_elem / (ranks_(d) * size_(d)), false,
                              true);

            arma::Mat<Real> U;
            arma::Col<Real> s;
            arma::Mat<Real> V;

            truncatedSvd<Real, Index>(C, deltaSquare, U, s, V, ranks_(d + 1));

            arma::Cube<Real> D(U.memptr(), ranks_(d), size_(d), ranks_(d + 1),
                               true, false);
            arrayCopy = arma::vectorise(arma::diagmat(s) * V.t());

            cores_(d).set_size(ranks_(d), ranks_(d + 1), size_(d));
            for (int k = 0; k < size_(d); ++k) {
                for (int j = 0; j < ranks_(d + 1); ++j) {
                    for (int i = 0; i < ranks_(d); ++i) {
                        cores_(d)(i, j, k) = D(i, k, j);
                    }
                }
            }
        } else {
            ranks_(d + 1) = 1;
            cores_(d) = arma::Cube<Real>(arrayCopy.memptr(), ranks_(d),
                                         ranks_(d + 1), size_(d), true, false);
        }
    }
}

template <typename Real, typename Index>
Real TtVector<Real, Index>::operator()(const arma::Col<Index> &index) const {
    arma::Mat<Real> temp;

    for (Index l = 0; l < ndim_; ++l) {
        Index d = ndim_ - 1 - l;

        if (d == ndim_ - 1) {
            temp = cores_(d).slice(index(d));
        } else {
            temp = cores_(d).slice(index(d)) * temp;
        }
    }

    return temp(0, 0);
}

template <typename Real, typename Index>
TtVector<Real, Index> TtVector<Real, Index>::operator+(
    const TtVector<Real, Index> &other) const {
    if (!arma::all(size_ == other.size_)) {
        throw std::logic_error(
            "TtVector::operator+() - Sizes of the TT vectors does not match");
    }

    // new ranks

    arma::Col<Index> ranks_new = ranks_ + other.ranks_;
    ranks_new(0) = 1;
    ranks_new(ndim_) = 1;

    // new cores

    arma::field<arma::Cube<Real>> cores_new(ndim_);

    // first core

    cores_new(0).zeros(1, ranks_new(1), size_(0));

    for (Index k = 0; k < size_(0); ++k) {
        for (Index j = 0; j < ranks_(1); ++j) {
            cores_new(0)(0, j, k) = cores_(0)(0, j, k);
        }

        for (Index j = 0; j < other.ranks_(1); ++j) {
            cores_new(0)(0, ranks_(1) + j, k) = other.cores_(0)(0, j, k);
        }
    }

    // middle cores

    for (Index d = 1; d < ndim_ - 1; ++d) {
        cores_new(d).zeros(ranks_new(d), ranks_new(d + 1), size_(d));

        for (Index k = 0; k < size_(d); ++k) {
            for (Index j = 0; j < ranks_(d + 1); ++j) {
                for (Index i = 0; i < ranks_(d); ++i) {
                    cores_new(d)(i, j, k) = cores_(d)(i, j, k);
                }
            }

            for (Index j = 0; j < other.ranks_(d + 1); ++j) {
                for (Index i = 0; i < other.ranks_(d); ++i) {
                    cores_new(d)(ranks_(d) + i, ranks_(d + 1) + j, k) =
                        other.cores_(d)(i, j, k);
                }
            }
        }
    }

    // last core

    cores_new(ndim_ - 1).zeros(ranks_new(ndim_ - 1), 1, size_(ndim_ - 1));

    for (Index k = 0; k < size_(ndim_ - 1); ++k) {
        for (Index i = 0; i < ranks_(ndim_ - 1); ++i) {
            cores_new(ndim_ - 1)(i, 0, k) = cores_(ndim_ - 1)(i, 0, k);
        }

        for (Index i = 0; i < other.ranks_(ndim_ - 1); ++i) {
            cores_new(ndim_ - 1)(ranks_(ndim_ - 1) + i, 0, k) =
                other.cores_(ndim_ - 1)(i, 0, k);
        }
    }

    return TtVector<Real, Index>(cores_new);
}

template <typename Real, typename Index>
TtVector<Real, Index> TtVector<Real, Index>::operator-(
    const TtVector<Real, Index> &other) const {
    return *this + other * (-1.0);
}

template <typename Real, typename Index>
TtVector<Real, Index> TtVector<Real, Index>::operator*(Real constant) const {
    arma::field<arma::Cube<Real>> cores_new(ndim_);

    for (Index d = 0; d < ndim_; ++d) {
        cores_new(d) = cores_(d);
    }

    cores_new(ndim_ - 1) *= constant;

    return TtVector<Real, Index>(cores_new);
}

template <typename Real, typename Index>
TtVector<Real, Index> TtVector<Real, Index>::operator/(Real constant) const {
    if (std::abs(constant) < std::numeric_limits<Real>::epsilon()) {
        throw std::logic_error(
            "TtVector::operator/() - Scalar is too close to zero");
    }

    return *this * (1 / constant);
}

template <typename Real, typename Index>
Real TtVector<Real, Index>::dot(const TtVector<Real, Index> &other) const {
    if (arma::any(size_ != other.size_)) {
        throw std::logic_error(
            "TtVector::dot() - Sizes of the two TT vectors does not match");
    }

    arma::Cube<Real> temp_3d;
    arma::Mat<Real> temp_2d;

    for (unsigned int l = 0; l < ndim_; ++l) {
        Index d = ndim_ - 1 - l;

        temp_3d.set_size(other.ranks_(d), ranks_(d), size_(d));

        if (d == ndim_ - 1) {
            // Kronecker product of the last cores
            for (Index k = 0; k < size_(d); ++k) {
                temp_3d.slice(k) =
                    other.cores_(d).slice(k) * cores_(d).slice(k).t();
            }
        } else {
            // multiplication by Kronecker product of the cores
            for (Index k = 0; k < size_(d); ++k) {
                temp_3d.slice(k) = other.cores_(d).slice(k) *
                                   (temp_2d * cores_(d).slice(k).t());
            }
        }

        temp_2d = arma::sum(temp_3d, 2);
    }

    return temp_2d(0, 0);
}

template <typename Real, typename Index>
Real TtVector<Real, Index>::norm2() const {
    return std::sqrt(this->dot(*this));
}

// non-member convenience functions --------------------------------------------

/// @name Arithmatic Operations
///@{

/// TT-vector scalar multiplication
template <typename Real, typename Index>
inline TtVector<Real, Index> operator*(Real constant,
                                       const TtVector<Real, Index> &vector) {
    return vector * constant;
}

///@}

/// @name Mathematical Functions
///@{

/// TT-vector dot
template <typename Real, typename Index>
inline Real dot(const TtVector<Real, Index> &vector1,
                const TtVector<Real, Index> &vector2) {
    return vector1.dot(vector2);
}

/// TT-vector 2-norm
template <typename Real, typename Index>
inline Real norm2(const TtVector<Real, Index> &vector) {
    return vector.norm2();
}

///@}

#endif
