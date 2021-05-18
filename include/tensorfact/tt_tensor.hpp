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
class TtTensor {
public:
    /// Default constructor
    TtTensor() = default;

    /// Construct a TT-tensor from the parameters
    TtTensor(long ndim, const std::vector<long> &size,
             const std::vector<long> &rank, const std::vector<double> &param);

    /// Default destructor
    ~TtTensor() = default;

    /// TT-ranks of the TT-tensor
    long Rank(long d) const { return rank_[d]; }

    /// Number of parameters
    long NumParam() const { return offset_[ndim_]; }

    /// Compute and return the entry of the TT-tensor at given index
    double Entry(const std::vector<long> &index) const;

    /// Write to file
    void WriteToFile(const std::string &file_name) const;

    /// Read from file
    void ReadFromFile(const std::string &file_name);

    /// Addition
    TtTensor operator+(const TtTensor &other) const;

    /// Subtraction
    TtTensor operator-(const TtTensor &other) const;

    /// Scalar multiplication
    TtTensor operator*(double alpha) const;

    /// Scalar division
    TtTensor operator/(double alpha) const;

    /// Rounding
    void Round(double relative_tolerance);

    /// Concatenation
    TtTensor Concatenate(const TtTensor &other, long dim, double rel_acc) const;

    /// Dot product
    double Dot(const TtTensor &other) const;

    /// 2-norm
    double FrobeniusNorm() const;

private:
    /// Linear index for unwrapping paramter vector
    long LinearIndex(long i, long j, long k, long d) const;

    /// Zero-padding to the back of a dimension
    TtTensor AddZeroPaddingBack(long dim, long pad) const;

    /// Zero-padding to the front of a dimension
    TtTensor AddZeroPaddingFront(long dim, long pad) const;

    long ndim_;
    std::vector<long> size_;
    std::vector<long> rank_;
    std::vector<long> offset_;
    std::vector<double> param_;
};

}  // namespace tensorfact

/// Scalar multiplication
inline tensorfact::TtTensor operator*(double alpha,
                                      const tensorfact::TtTensor &tensor) {
    return tensor * alpha;
}

#endif
