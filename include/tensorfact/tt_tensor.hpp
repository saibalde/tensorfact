/// @file tt_tensor.hpp

#ifndef TENSORFACT_TTTENSOR_HPP
#define TENSORFACT_TTTENSOR_HPP

#include <string>
#include <vector>

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

    /// Construct a TT-tensor from the parameters
    TtTensor(long ndim, const std::vector<long> &size,
             const std::vector<long> &rank, const std::vector<Real> &param);

    /// Default destructor
    ~TtTensor() = default;

    /// Default copy constructor
    TtTensor(const TtTensor<Real> &) = default;

    /// Default copy assignment
    TtTensor<Real> &operator=(const TtTensor<Real> &) = default;

    /// Default move constructor
    TtTensor(TtTensor<Real> &&) = default;

    /// Default move assignment
    TtTensor<Real> &operator=(TtTensor<Real> &&) = default;

    /// Number of dimensions
    const long &NumDim() const { return ndim_; }

    /// Mode sizes
    const std::vector<long> &Size() const { return size_; }

    /// Number of elements in full tensor
    long NumElement() const;

    /// TT-ranks
    const std::vector<long> &Rank() const { return rank_; }

    /// Number of parameters
    const long &NumParam() const { return offset_[ndim_]; }

    /// Parameters
    const std::vector<Real> &Param() const { return param_; }

    /// Parameter
    const Real &Param(long i, long j, long k, long d) const {
        return param_[LinearIndex(i, j, k, d)];
    }

    /// Parameter
    Real &Param(long i, long j, long k, long d) {
        return param_[LinearIndex(i, j, k, d)];
    }

    /// In-place addition
    TtTensor<Real> operator+=(const TtTensor<Real> &other);

    /// In-place subtraction
    TtTensor<Real> operator-=(const TtTensor<Real> &other);

    /// In-place scalar multiplication
    TtTensor<Real> operator*=(Real alpha);

    /// In-place scalar division
    TtTensor<Real> operator/=(Real alpha);

    /// In-place elementwise multiplication
    TtTensor<Real> operator*=(const TtTensor<Real> &other);

    /// Addition
    TtTensor<Real> operator+(const TtTensor<Real> &other) const;

    /// Subtraction
    TtTensor<Real> operator-(const TtTensor<Real> &other) const;

    /// Scalar multiplication
    TtTensor<Real> operator*(Real alpha) const;

    /// Scalar division
    TtTensor<Real> operator/(Real alpha) const;

    /// Elementwise multiplication
    TtTensor<Real> operator*(const TtTensor<Real> &other) const;

    /// Shift
    TtTensor<Real> Shift(long site, long shift) const;

    /// Dot product
    Real Dot(const TtTensor<Real> &other) const;

    /// 2-norm
    Real FrobeniusNorm() const;

    /// Rounding
    void Round(Real relative_tolerance);

    /// Concatenation
    TtTensor<Real> Concatenate(const TtTensor<Real> &other, long dim,
                               Real relative_tolerance) const;

    /// Compute and return the entry of the TT-tensor at given index
    Real Entry(const std::vector<long> &index) const;

    /// Convert to full
    std::vector<Real> Full() const;

private:
    /// Linear index for unwrapping paramter vector
    long LinearIndex(long i, long j, long k, long d) const {
        return i + rank_[d] * (j + size_[d] * k) + offset_[d];
    }

    /// Zero-padding to the back of a dimension
    TtTensor<Real> AddZeroPaddingBack(long dim, long pad) const;

    /// Zero-padding to the front of a dimension
    TtTensor<Real> AddZeroPaddingFront(long dim, long pad) const;

    long ndim_;
    std::vector<long> size_;
    std::vector<long> rank_;
    std::vector<long> offset_;
    std::vector<Real> param_;
};

/// Scalar multiplication
template <typename Real>
inline TtTensor<Real> operator*(Real alpha, const TtTensor<Real> &tensor) {
    return tensor * alpha;
}

}  // namespace tensorfact

#endif
