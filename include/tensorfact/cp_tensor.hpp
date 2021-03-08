/// @file cp_tensor.hpp

#ifndef TENSORFACT_CPTENSOR_HPP
#define TENSORFACT_CPTENSOR_HPP

#include <armadillo>
#include <string>

namespace tensorfact {

/// @brief CP representation of a multidimensional tensor
///
/// A CP tensor is a memory-efficient representation of a multidimensional array
/// \f$\mathcal{X}(i_0, \ldots, i_{d - 1})\f$ where the entries are computed as
/// \f[
/// \mathcal{X}(i_0, \ldots, i_{d - 1}) = \sum_{\alpha = 0}^{r - 1}
/// \mathbf{X}_0(i_0, \alpha) \cdots \mathbf{X}_{d - 1}(i_{d - 1}, \alpha)
/// \f]
/// Assuming \f$n_k \sim n\f$, this reduces the storage complexity
/// \f$\mathcal{O}(n^d)\f$ of the full tensor to \f$\mathcal{O}(d n r)\f$ in the
/// CP format.
template <typename Real>
class CpTensor {
public:
    /// Default constructor
    CpTensor() = default;

    /// Construct a CP tensor from its factors
    CpTensor(const arma::field<arma::Mat<Real>> &factor);

    /// Default destructor
    ~CpTensor() = default;

    /// Compute and return entry of CP tensor at specified index
    Real operator()(const arma::Col<arma::uword> &index) const;

    /// Write to file
    void WriteToFile(const std::string &file_name) const;

    /// Read from file
    void ReadFromFile(const std::string &file_name);

private:
    arma::uword ndim_;
    arma::Col<arma::uword> size_;
    arma::uword rank_;
    arma::field<arma::Mat<Real>> factor_;
};

}  // namespace tensorfact

#endif
