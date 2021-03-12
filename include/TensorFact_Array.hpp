/// @file TensorFact_Array.hpp

#ifndef TENSORFACT_ARRAY_HPP
#define TENSORFACT_ARRAY_HPP

#include <vector>

namespace TensorFact {

/// @brief Wrapper around std::vector to support multidimensional arrays
template <typename Scalar>
class Array {
public:
    /// Default constructor
    Array() = default;

    /// Default destructor
    ~Array() = default;

    /// Dimensionality
    std::size_t NDim() const { return size_.size(); }

    /// Size
    const std::vector<std::size_t> &Size() const { return size_; }

    /// Size along a dimension
    std::size_t Size(std::size_t dim) const { return size_[dim]; }

    /// Total number of elements
    std::size_t NumberOfElements() const { return unfolding_factors_[ndim_]; }

    /// Entry at cartesian index
    const Scalar &operator()(
        const std::vector<std::size_t> &cartesian_index) const;

    /// Entry at cartesian index
    Scalar &operator()(const std::vector<std::size_t> &cartesian_index);

    /// Entry at linear index
    const Scalar &operator()(std::size_t linear_index) const {
        return entries_[linear_index];
    }

    /// Entry at linear index
    Scalar &operator()(std::size_t linear_index) {
        return entries_[linear_index];
    }

    /// Addition
    Array<Scalar> operator+(const Array<Scalar> &other) const;

    /// Scalar multiplication
    Array<Scalar> operator*(Scalar alpha) const;

    /// Subtraction
    Array<Scalar> operator-(const Array<Scalar> &other) const;

    /// Scalar division
    Array<Scalar> operator/(const Scalar alpha) const;

    /// Reshape
    void Reshape(const std::vector<std::size_t> &size);

    /// Resize
    ///
    /// @warning Destroys data
    void Resize(const std::vector<std::size_t> &size);

    /// Frobenius norm
    Scalar FrobeniusNorm() const;

    /// Multiplication
    ///
    /// @note Only implemented for matrix-vector and matrix-matrix cases
    ///
    /// For arrays `A` and `B` with `A.NDim() == 2` and `B.NDim() == 1`,
    /// `A.Multiply(conjugate_A, B, conjuate_B, C)` ignores the value of
    /// `conjugate_B` and computes
    /// - `C = A * B` if `conjugate_A == false`
    /// - `C = A^H * B` if `conjuage_A == true`
    ///
    /// For arrays `A` and `B` with `A.NDim() == 2` and `B.NDim() == 2`,
    /// `A.Muliply(conjugate_A, B, conjugate_B, C)` computes
    /// - `C = A * B` if `conjugate_A == false` and `conjugate_B == false`
    /// - `C = A^H * B` if `conjugate_A == true` and `conjugate_B == false`
    /// - `C = A * B^H` if `conjugate_A == false` and `conjugate_B == true`
    /// - `C = A^H * B^H` if `conjugate_A == true` and `conjugate_B == true`
    void Multiply(bool conjugate, const Array<Scalar> &other,
                  bool other_conjugate, Array<Scalar> &result) const;

    /// RQ orthogonalization
    ///
    /// For array `A` with `A.Size(0) == m` and `A.Size(1) == n`,
    /// `A.ReducedRq(R, Q)` computes the RQ factorization satisfying `A == R *
    /// Q`.
    ///
    /// @note Only implemented for matrices
    void ReducedRq(Array<Scalar> &R, Array<Scalar> &Q) const;

    /// Truncated singular value decomposition
    ///
    /// For array `A` with `A.Size(0) == m` and `A.Size(1) == n`,
    /// `A.ReducedSvd(U, s, Vt, tol, relative_flag)` computes the truncated
    /// singular value decomposition with minimum possible rank satisfying
    /// - `(A - U * diag(s) * Vt).FrobeniusNorm() <= tol` if `relative_flag ==
    ///   false`
    /// - `(A - U * diag(s) * Vt).FrobeniusNorm() <= tol * A.FrobeniusNorm()` if
    ///   `relative_flag == true`
    ///
    /// @note Only implemented for matrices
    ///
    /// @note For `tol <= machine_precision` this computes the thin SVD
    void TruncatedSvd(Array<Scalar> &U, Array<Scalar> &s, Array<Scalar> &Vt,
                      Scalar tolernace, bool relative_flag) const;

private:
    /// Cartesian-to-linear index conversion
    void CartesianToLinearIndex(const std::vector<std::size_t> &cartesian_index,
                                std::size_t &linear_index) const;

    /// Linear-to-Cartesian index conversion
    void LinearToCartesianIndex(
        std::size_t linear_index,
        std::vector<std::size_t> &cartesian_index) const;

    std::size_t ndim_;
    std::vector<std::size_t> size_;
    std::vector<std::size_t> unfolding_factors_;
    std::vector<Scalar> entries_;
};

}  // namespace TensorFact

/// Scalar multiplication
template <typename Scalar>
inline TensorFact::Array<Scalar> operator*(
    Scalar alpha, const TensorFact::Array<Scalar> &array) {
    return array * alpha;
}

#endif
