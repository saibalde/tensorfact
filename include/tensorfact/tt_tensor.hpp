/// @file tt_tensor.hpp

#ifndef TENSORFACT_TTTENSOR_HPP
#define TENSORFACT_TTTENSOR_HPP

#include <string>
#include <vector>

namespace tensorfact {

/// @brief TT representation of a multidimensional tensor
///
/// A TT-tensor is a memory-efficient representation of a multidimensional array
/// \f$\mathcal{X}(i_0, \ldots, i_{d - 1})\f$ for \f$1 \leq i_k \leq n_k\f$ and
/// \f$0 \leq k \leq d - 1\f$, where the entries are computed as
/// \f[
/// \mathcal{X}(i_0, \ldots, i_{d - 1}) = X_0(i_0) \cdots X_{d - 1}(i_{d - 1})
/// \f]
/// Here \f$X_k(i_k) \in \mathbb{R}^{r_{k - 1} \times r_k}\f$ is the
/// \f$i_k\f$-th slice of the \f$k\f$-th TT core \f$\mathcal{X}_k \in
/// \mathbb{R}^{r_{k - 1} \times n_k \times r_k}\f$:
/// \f[
/// X_k(i_k) = \mathcal{X}_k(:, i_k, :)
/// \f]
/// The numbers \f$\{r_0, \ldots, r_{d - 1}\}\f$ are called the TT ranks; to
/// satisfy the properties of matrix multiplication, we require \f$r_0 = r_d =
/// 1\f$.
///
/// @note Implemented for template parameter `Real = float` and `Real = double`
template <typename Real>
class TtTensor {
public:
    ///@{

    /// @brief Default constructor
    TtTensor() = default;

    /// @brief Construct a TT-tensor from ranks and sizes with undefined
    /// parameters
    ///
    /// Given number of dimensions \f$d\f$, mode size \f$n\f$ and interior TT
    /// rank \f$r\f$, constructs a \f$d\f$-dimensional TT tensor with
    /// \f$n_k = n\f$ and \f$r_k = r\f$ for \f$1 \leq k \leq d - 1\f$; boundary
    /// TT ranks \f$r_0\f$ and \f$r_d\f$ are set to one. Entries of the TT cores
    /// are undefined.
    ///
    /// @param [in] num_dim Number of dimensions \f$d\f$
    /// @param [in] size    Mode size \f$n\f$
    /// @param [in] rank    Interior TT rank \f$r\f$
    TtTensor(long num_dim, long size, long rank);

    /// @brief Construct a TT-tensor from ranks and sizes with undefined
    /// parameters
    ///
    /// Given number of dimensions \f$d\f$, mode sizes \f$\{n_0, \ldots,
    /// n_{d - 1}\}\f$ and TT ranks \f$\{r_0, \ldots, r_d\}\f$ constructs the
    /// corresponding TT tensor. Entries of the TT cores are undefined.
    ///
    /// @param [in] num_dim Number of dimensions
    /// @param [in] size    Mode sizes \f$\{n_0, \ldots, n_{d - 1}\}\f$
    /// @param [in] rank    TT ranks \f$\{r_0, \ldots, r_d\}\f$
    TtTensor(long num_dim, const std::vector<long> &size,
             const std::vector<long> &rank);

    /// @brief Construct a TT-tensor from ranks, sizes and parameters
    ///
    /// Given number of dimensions \f$d\f$, mode sizes \f$\{n_0, \ldots,
    /// n_{d - 1}\}\f$, TT ranks \f$\{r_0, \ldots, r_d\}\f$ and parameter value
    /// vector \f$v\f$ constructs the corresponding TT tensor. Entries of the
    /// TT cores are selected from \f$v\f$ using a column-major unwrapping:
    /// \f[
    /// \mathcal{X}_k(\alpha_k, i_k, \alpha_{k + 1}) = v(i), \quad i = \alpha_k
    /// + i_k r_k + \alpha_{k + 1} r_k n_k + \sum_{j = 0}^{k - 1} r_j n_j
    /// r_{j + 1}
    /// \f]
    ///
    /// @param [in] num_dim Number of dimensions
    /// @param [in] size    Mode sizes \f$\{n_0, \ldots, n_{d - 1}\}\f$
    /// @param [in] rank    TT ranks \f$\{r_0, \ldots, r_d\}\f$
    /// @param [in] param   Parameter value vector \f$v\f$
    TtTensor(long num_dim, const std::vector<long> &size,
             const std::vector<long> &rank, const std::vector<Real> &param);

    /// @brief Default copy constructor
    TtTensor(const TtTensor<Real> &) = default;

    /// @brief Default move constructor
    TtTensor(TtTensor<Real> &&) = default;

    /// @brief Default destructor
    ~TtTensor() = default;

    ///@}

    ///@{

    /// @brief Default copy assignment
    TtTensor<Real> &operator=(const TtTensor<Real> &) = default;

    /// @brief Default move assignment
    TtTensor<Real> &operator=(TtTensor<Real> &&) = default;

    ///@}

    ///@{

    /// @brief Number of dimensions
    ///
    /// @return Dimensionality \f$d\f$ of TT tensor
    const long &NumDim() const { return num_dim_; }

    /// @brief Mode sizes
    ///
    /// @return Vector of mode sizes \f$\{n_0, \ldots, n_{d - 1}\}\f$ of TT
    /// tensor
    const std::vector<long> &Size() const { return size_; }

    /// @brief Mode size
    ///
    /// @param [in] d   Dimension index \f$k\f$
    //
    /// @return Mode size \f$n_k\f$ of TT tensor
    const long &Size(long d) const { return size_[d]; }

    /// @brief TT-ranks
    ///
    /// @return Vector of TT ranks \f$\{r_0, \ldots, r_d\}\f$ of TT tensor
    const std::vector<long> &Rank() const { return rank_; }

    /// @brief TT-rank
    ///
    /// @param [in] d   Dimension index \f$k\f$
    ///
    /// @return TT rank \f$r_k\f$ of TT tensor
    const long &Rank(long d) const { return rank_[d]; }

    /// @brief Parameters of TT tensor (TT core entries)
    ///
    /// @return Vector of TT core entries, unwrapped in column-major fashion
    const std::vector<Real> &Param() const { return param_; }

    /// @brief Parameter of TT tensor (TT core entry)
    ///
    /// @param [in] i   Rank index \f$\alpha_k\f$
    /// @param [in] j   Slice index \f$i_k\f$
    /// @param [in] k   Rank index \f$\alpha_{k + 1}\f$
    /// @param [in] d   Dimension index \f$k\f$
    ///
    /// @return Entry \f$\mathcal{X}_k(\alpha_k, i_k, \alpha_{k + 1})\f$ of
    /// \f$k\f$-th TT core
    const Real &Param(long i, long j, long k, long d) const {
        return param_[LinearIndex(i, j, k, d)];
    }

    /// @brief Number of parameters
    ///
    /// @return Total number of parameters (entries of all TT cores)
    const long &NumParam() const { return offset_[num_dim_]; }

    /// @brief Number of elements in full tensor
    ///
    /// @return Total number of elements in uncompressed tensor
    long NumElement() const;

    /// @brief Parameters of TT tensor (TT core entries)
    ///
    /// @return Modifiable reference to vector of TT core entries, unwrapped in
    /// column-major fashion
    std::vector<Real> &Param() { return param_; }

    /// @brief Parameter of TT tensor (TT core entry)
    ///
    /// @param [in] i   Rank index \f$\alpha_k\f$
    /// @param [in] j   Slice index \f$i_k\f$
    /// @param [in] k   Rank index \f$\alpha_{k + 1}\f$
    /// @param [in] d   Dimension index \f$k\f$
    ///
    /// @return Modifiable reference to entry \f$\mathcal{X}_k(\alpha_k, i_k,
    /// \alpha_{k + 1})\f$ of \f$k\f$-th TT core
    Real &Param(long i, long j, long k, long d) {
        return param_[LinearIndex(i, j, k, d)];
    }

    ///@}

    ///@{

    /// @brief In-place addition with TT tensor
    ///
    /// Allows \f$\mathcal{X} += \mathcal{Y}\f$
    ///
    /// @param [in] other   Other tensor \f$\mathcal{Y}\f$
    ///
    /// @return Updated \f$\mathcal{X}\f$
    TtTensor<Real> operator+=(const TtTensor<Real> &other);

    /// @brief In-place subtraction of TT tensor
    ///
    /// Allows \f$\mathcal{X} -= \mathcal{Y}\f$
    ///
    /// @param [in] other   Other tensor \f$\mathcal{Y}\f$
    ///
    /// @return Updated \f$\mathcal{X}\f$
    TtTensor<Real> operator-=(const TtTensor<Real> &other);

    /// @brief In-place multiplication by scalar
    ///
    /// Allows \f$\mathcal{X} *= \alpha\f$
    ///
    /// @param [in] alpha   Scalar \f$\alpha\f$
    ///
    /// @return Updated \f$\mathcal{X}\f$
    TtTensor<Real> operator*=(Real alpha);

    /// @brief In-place division by scalar
    ///
    /// Allows \f$\mathcal{X} /= \alpha\f$
    ///
    /// @param [in] alpha   Scalar \f$\alpha\f$
    ///
    /// @return Updated \f$\mathcal{X}\f$
    TtTensor<Real> operator/=(Real alpha);

    /// @brief In-place elementwise multiplication by TT tensor
    ///
    /// Allows \f$\mathcal{X} \circ= \mathcal{Y}\f$ where \f$\circ\f$ is
    /// elementwise multiplication
    ///
    /// @param [in] other   Other tensor \f$\mathcal{Y}\f$
    ///
    /// @return Updated \f$\mathcal{X}\f$
    TtTensor<Real> operator*=(const TtTensor<Real> &other);

    ///@}

    ///@{

    /// @brief Addition of two TT tensors
    ///
    /// Allows \f$\mathcal{Z} = \mathcal{X} + \mathcal{Y}\f$ where
    /// \f$\mathcal{X}\f$ is this tensor, \f$\mathcal{Y}\f$ is other tensor.
    ///
    /// @param [in] other   Other tensor \f$\mathcal{Y}\f$
    ///
    /// @return Result tensor \f$\mathcal{Z}\f$
    TtTensor<Real> operator+(const TtTensor<Real> &other) const;

    /// @brief Subtraction of two TT tensors
    ///
    /// Allows \f$\mathcal{Z} = \mathcal{X} - \mathcal{Y}\f$ where
    /// \f$\mathcal{X}\f$ is this tensor, \f$\mathcal{Y}\f$ is other tensor.
    ///
    /// @param [in] other   Other tensor \f$\mathcal{Y}\f$
    ///
    /// @return Result tensor \f$\mathcal{Z}\f$
    TtTensor<Real> operator-(const TtTensor<Real> &other) const;

    /// @brief Multiplication of TT tensor with scalar
    ///
    /// Allows \f$\mathcal{Z} = \mathcal{X} * \alpha\f$ where \f$\mathcal{X}\f$
    /// is this tensor, \f$\alpha\f$ is scalar.
    ///
    /// @param [in] alpha   Scalar \f$\alpha\f$
    ///
    /// @return Result tensor \f$\mathcal{Z}\f$
    TtTensor<Real> operator*(Real alpha) const;

    /// @brief Division of TT tensor by scalar
    ///
    /// Allows \f$\mathcal{Z} = \mathcal{X} / \alpha\f$ where \f$\mathcal{X}\f$
    /// is this tensor, \f$\alpha\f$ is scalar.
    ///
    /// @param [in] alpha   Scalar \f$\alpha\f$
    ///
    /// @return Result tensor \f$\mathcal{Z}\f$
    TtTensor<Real> operator/(Real alpha) const;

    /// @brief Elementwise multiplication of TT tensor by another TT tensor
    ///
    /// Allows \f$\mathcal{Z} = \mathcal{X} \circ \mathcal{Y}\f$ where
    /// \f$\mathcal{X}\f$ is this tensor, \f$\mathcal{Y}\f$ is other tensor and
    /// \f$\circ\f$ is elementwise multiplication.
    ///
    /// @param [in] other   Other tensor \f$\mathcal{Y}\f$
    ///
    /// @return Result tensor \f$\mathcal{Z}\f$
    TtTensor<Real> operator*(const TtTensor<Real> &other) const;

    ///@}

    ///@{

    /// @brief Concatenation of two TT tensors
    ///
    /// Given this tensor \f$\mathcal{X}\f$ and other tensor \f$\mathcal{Y}\f$
    /// with same mode sizes except along the \f$k\f$-th dimension, construct
    /// a new tensor \f$\mathcal{Z}\f$ such that
    /// \f[
    /// \mathcal{Z}(\ldots, i_k, \ldots) =
    /// \begin{cases}
    /// \mathcal{X}(\ldots, i_k, \ldots) & \text{if} \quad 0 \leq i_k <
    /// n_k(\mathcal{X}) \\
    /// \mathcal{Y}(\ldots, i_k - n_k(\mathcal{X}), \ldots) & \text{if} \quad
    /// n_k(\mathcal{X}) \leq i_k < n_k(\mathcal{X}) + n_k(\mathcal{Y})
    /// \end{cases}
    /// \f]
    ///
    /// @param [in] other   Other TT tensor \f$\mathcal{Y}\f$
    /// @param [in] dim     Dimension index \f$k\f$
    ///
    /// @return Concatenated tensor \f$\mathcal{Z}\f$
    TtTensor<Real> Concatenate(const TtTensor<Real> &other, long dim,
                               Real relative_tolerance) const;

    /// @brief Shift of entries along dimension
    ///
    /// Given dimension index \f$k\f$ and shift \f$l\f$ construct a TT tensor
    /// \f$\mathcal{Z}\f$ same size such that
    /// \f[
    /// \mathcal{Z}(\ldots, i_k, \ldots) = \mathcal{X}(\ldots, i_k + l, \ldots)
    /// \f]
    /// when right hand side is known; rest of the entries are set to zero
    ///
    /// @param [in] site    Dimension index \f$k\f$
    /// @param [in] shift   Shift \f$l\f$
    ///
    /// @return Shifted tensor \f$\mathcal{Z}\f$
    TtTensor<Real> Shift(long site, long shift) const;

    ///@}

    ///@{

    /// @brief Contraction
    ///
    /// Contraction of this tensor \f$\mathcal{X}\f$ with vectors \f$\{v_0,
    /// \ldots, v_{d - 1}\}\f$, computing the quantity
    /// \f[
    /// \alpha = \sum_{i_0 = 0}^{n_0 - 1} \cdots
    /// \sum_{i_{d - 1} = 0}^{n_{d - 1} - 1} \mathcal{X}(i_0, \cdots, i_{d - 1})
    /// v_0(i_0) \cdots v_{d - 1}(i_{d - 1})
    /// \f]
    ///
    /// @param [in] vectors The vectors \f$\{v_0, \ldots, v_{d - 1}\}\f$
    ///
    /// @return Contraction value \f$\alpha\f$
    Real Contract(const std::vector<std::vector<Real>> &vectors) const;

    ///@}

    ///@{

    /// @brief Dot product
    ///
    /// Dot product of this tensor \f$\mathcal{X}\f$ and other tensor
    /// \f$\mathcal{Y}\f$ of the same size \f$n_0 \times \cdots \times
    /// n_{d - 1}\f$
    /// \f[
    /// \alpha = \sum_{i_0 = 0}^{n_0 - 1} \cdots
    /// \sum_{i_{d - 1} = 0}^{n_{d - 1} - 1} \mathcal{X}(i_0, \ldots, i_{d - 1})
    /// \mathcal{Y}(i_0, \ldots, i_{d - 1})
    /// \f]
    ///
    /// @param [in] other   Other tensor \f$\mathcal{Y}\f$
    ///
    /// @return Dot product \f$\alpha\f$
    Real Dot(const TtTensor<Real> &other) const;

    /// @brief 2-norm
    ///
    /// Frobenius norm of this tensor \f$\mathcal{X}\f$ of size \f$n_0 \times
    /// \cdots \times n_{d - 1}\f$
    /// \f[
    /// \Vert \mathcal{X} \Vert_2 = \left(\sum_{i_0 = 0}^{n_0 - 1} \cdots
    /// \sum_{i_{d - 1} = 0}^{n_{d - 1} - 1} \mathcal{X}(i_0, \ldots,
    /// i_{d - 1})^2\right)^{1/2}
    /// \f]
    ///
    /// @return Frobenius norm \f$\Vert \mathcal{X} \Vert_2\f$
    Real FrobeniusNorm() const;

    ///@}

    ///@{

    /// @brief Rounding
    ///
    /// Optimize the ranks of this TT tensor \f$\mathcal{X}\f$ with specified
    /// relative tolerace \f$\tau\f$ such that \f$\Vert \mathcal{X}_\text{new}
    /// - \mathcal{X} \Vert_2 \leq \tau \Vert \mathcal{X} \Vert_2\f$
    ///
    /// @param [in] relative_tolerance  Tolerance \f$\tau\f$
    void Round(Real relative_tolerance);

    ///@}

    ///@{

    /// @brief Convert to full
    ///
    /// Convert TT tensor to full tensor
    ///
    /// @return Full tensor in column-major order
    std::vector<Real> Full() const;

    ///@}

    ///@{

    /// @brief TT-tensor entry
    ///
    /// Compute the entry of the TT-tensor \f$\mathcal{X}\f$ at index
    /// \f$(i_0, \ldots, i_{d - 1})\f$
    ///
    /// @param [in] index Index vector \f$(i_0, \ldots, i_{d - 1})\f$
    ///
    /// @return Tensor entry \f$\mathcal{X}(i_0, \ldots, i_{d - 1})\f$
    Real Entry(const std::vector<long> &index) const;

    ///@}

    ///@{

    /// @brief Write TT-Tensor to text file
    ///
    /// @param [in] file_name   Name of file where TT tensor will be saved
    void WriteToFile(const std::string &file_name) const;

    /// @brief Read TT-Tensor from text file
    ///
    /// @param [in] file_name   Name of file from where TT tensor will be read
    void ReadFromFile(const std::string &file_name);

    ///@}

private:
    /// Linear index for unwrapping paramter vector
    long LinearIndex(long i, long j, long k, long d) const {
        return i + rank_[d] * (j + size_[d] * k) + offset_[d];
    }

    /// Zero-padding to the back of a dimension
    TtTensor<Real> AddZeroPaddingBack(long dim, long pad) const;

    /// Zero-padding to the front of a dimension
    TtTensor<Real> AddZeroPaddingFront(long dim, long pad) const;

    long num_dim_;
    std::vector<long> size_;
    std::vector<long> rank_;
    std::vector<long> offset_;
    std::vector<Real> param_;
};

/// @brief Multiplication of TT tensor with scalar
///
/// Allows \f$\mathcal{Z} = \alpha * \mathcal{X}\f$ where \f$\mathcal{X}\f$
/// is a TT tensor, \f$\alpha\f$ is scalar.
///
/// @param [in] alpha   Scalar \f$\alpha\f$
/// @param [in] tensor  Tensor \f$\mathcal{X}\f$
///
/// @return Result tensor \f$\mathcal{Z}\f$
template <typename Real>
inline TtTensor<Real> operator*(Real alpha, const TtTensor<Real> &tensor) {
    return tensor * alpha;
}

}  // namespace tensorfact

#endif
