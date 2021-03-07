/// @file tt_tensor.hpp

#ifndef TENSORFACT_TTTENSOR_HPP
#define TENSORFACT_TTTENSOR_HPP

#include <armadillo>
#include <cmath>

namespace tensorfact {

/// @brief TT representation of a multidimensional tensor
///
/// A TT-tensor is a memory-efficient representation of a multidimensional array
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
template <typename Real>
class TtTensor {
public:
    /// Default constructor
    TtTensor() = default;

    /// Construct a TT-tensor from the cores
    TtTensor(const arma::field<arma::Cube<Real>> &cores);

    /// Construct a TT-tensor from full tensor using TT-SVD
    TtTensor(const arma::Col<Real> &array, const arma::Col<arma::uword> &size,
             Real rel_acc);

    /// Default destructor
    ~TtTensor() = default;

    /// Return the dimensionality of the TT-tensor
    const arma::uword &NDim() const { return ndim_; }

    /// Return the size of the TT-tensor
    const arma::Col<arma::uword> &Size() const { return size_; }

    /// Return the TT-ranks of the TT-tensor
    const arma::Col<arma::uword> &Rank() const { return rank_; }

    /// Return the specified core of the TT-tensor
    const arma::Cube<Real> &Core(arma::uword i) const { return core_(i); }

    /// Compute and return the entry of the TT-tensor at given index
    Real operator()(const arma::Col<arma::uword> &index) const;

    /// Addition
    TtTensor<Real> operator+(const TtTensor<Real> &other) const;

    /// Subtraction
    TtTensor<Real> operator-(const TtTensor<Real> &other) const;

    /// Scalar multiplication
    TtTensor<Real> operator*(Real alpha) const;

    /// Scalar division
    TtTensor<Real> operator/(Real alpha) const;

    /// Dot product
    Real Dot(const TtTensor<Real> &other) const;

    /// 2-norm
    Real Norm2() const { return std::sqrt(this->Dot(*this)); }

    /// Rounding
    TtTensor<Real> Round(Real rel_acc) const;

    /// Zero-padding to the back of a dimension
    TtTensor<Real> AddZeroPaddingBack(arma::uword dim, arma::uword pad) const;

    /// Zero-padding to the front of a dimension
    TtTensor<Real> AddZeroPaddingFront(arma::uword dim, arma::uword pad) const;

private:
    arma::uword ndim_;
    arma::Col<arma::uword> size_;
    arma::Col<arma::uword> rank_;
    arma::field<arma::Cube<Real>> core_;
};

}  // namespace tensorfact

/// Scalar multiplication
template <typename Real>
inline tensorfact::TtTensor<Real> operator*(
    Real alpha, const tensorfact::TtTensor<Real> &tensor) {
    return tensor * alpha;
}

#endif
