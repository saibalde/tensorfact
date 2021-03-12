/// @file TensorFact_TtTensor.hpp

#ifndef TENSORFACT_TTTENSOR_HPP
#define TENSORFACT_TTTENSOR_HPP

#include <cmath>
#include <string>
#include <vector>

#include "TensorFact_Array.hpp"

namespace TensorFact {

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
template <typename Scalar>
class TtTensor {
public:
    /// Default constructor
    TtTensor() = default;

    /// Construct a TT-tensor from the cores
    TtTensor(const std::vector<TensorFact::Array<Scalar>> &cores);

    /// Default destructor
    ~TtTensor() = default;

    /// Return the dimensionality of the TT-tensor
    const std::size_t &NDim() const { return ndim_; }

    /// Return the size of the TT-tensor
    const std::vector<std::size_t> &Size() const { return size_; }

    /// Return the TT-ranks of the TT-tensor
    const std::vector<std::size_t> &Rank() const { return rank_; }

    /// Return the specified core of the TT-tensor
    const TensorFact::Array<Scalar> &Core(std::size_t i) const {
        return core_[i];
    }

    /// Compute and return the entry of the TT-tensor at given index
    Scalar operator()(const std::vector<std::size_t> &index) const;

    /// Write to file
    void WriteToFile(const std::string &file_name) const;

    /// Read from file
    void ReadFromFile(const std::string &file_name);

    /// Addition
    TtTensor<Scalar> operator+(const TtTensor<Scalar> &other) const;

    /// Subtraction
    TtTensor<Scalar> operator-(const TtTensor<Scalar> &other) const;

    /// Scalar multiplication
    TtTensor<Scalar> operator*(Scalar alpha) const;

    /// Scalar division
    TtTensor<Scalar> operator/(Scalar alpha) const;

    /// Compute from full tensor using TT-SVD
    void ComputeFromFull(const TensorFact::Array<Scalar> &array,
                         Scalar rel_acc);

    /// Rounding
    void Round(Scalar rel_acc);

    /// Concatenation
    TtTensor<Scalar> Concatenate(const TtTensor<Scalar> &other, std::size_t dim,
                                 Scalar rel_acc) const;

    /// Dot product
    Scalar Dot(const TtTensor<Scalar> &other) const;

    /// 2-norm
    Scalar FrobeniusNorm() const { return std::sqrt(this->Dot(*this)); }

private:
    /// Zero-padding to the back of a dimension
    TtTensor<Scalar> AddZeroPaddingBack(std::size_t dim, std::size_t pad) const;

    /// Zero-padding to the front of a dimension
    TtTensor<Scalar> AddZeroPaddingFront(std::size_t dim,
                                         std::size_t pad) const;

    std::size_t ndim_;
    std::vector<std::size_t> size_;
    std::vector<std::size_t> rank_;
    std::vector<TensorFact::Array<Scalar>> core_;
};

}  // namespace TensorFact

/// Scalar multiplication
template <typename Scalar>
inline TensorFact::TtTensor<Scalar> operator*(
    Scalar alpha, const TensorFact::TtTensor<Scalar> &tensor) {
    return tensor * alpha;
}

#endif
