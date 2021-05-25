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
    /// Scalar type
    typedef Real scalar_type;

    /// Default constructor
    TtTensor() = default;

    /// Construct a TT-tensor from the parameters
    TtTensor(long ndim, const std::vector<long> &size,
             const std::vector<long> &rank, const std::vector<Real> &param);

    /// Default destructor
    ~TtTensor() = default;

    /// Number of dimensions
    long NumDim() const { return ndim_; }

    /// Mode size
    long Size(long d) const { return size_[d]; }

    /// TT-ranks of the TT-tensor
    long Rank(long d) const { return rank_[d]; }

    /// Number of elements
    long NumElement() const;

    /// Number of parameters
    long NumParam() const { return offset_[ndim_]; }

    /// Compute and return the entry of the TT-tensor at given index
    Real Entry(const std::vector<long> &index) const;

    /// Write to file
    void WriteToFile(const std::string &file_name) const;

    /// Read from file
    void ReadFromFile(const std::string &file_name);

    /// Addition
    TtTensor<Real> operator+(const TtTensor<Real> &other) const;

    /// Subtraction
    TtTensor<Real> operator-(const TtTensor<Real> &other) const;

    /// Scalar multiplication
    TtTensor<Real> operator*(Real alpha) const;

    /// Scalar division
    TtTensor<Real> operator/(Real alpha) const;

    /// Rounding
    void Round(Real relative_tolerance);

    /// Concatenation
    TtTensor<Real> Concatenate(const TtTensor<Real> &other, long dim,
                               Real relative_tolerance) const;

    /// Dot product
    Real Dot(const TtTensor<Real> &other) const;

    /// 2-norm
    Real FrobeniusNorm() const;

    /// Convert to full
    std::vector<Real> Full() const;

private:
    /// Linear index for unwrapping paramter vector
    long LinearIndex(long i, long j, long k, long d) const;

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
