/// @file TensorFact_CpTensor.hpp

#ifndef TENSORFACT_CPTENSOR_HPP
#define TENSORFACT_CPTENSOR_HPP

#include <string>
#include <vector>

#include "TensorFact_Array.hpp"

namespace TensorFact {

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
template <typename Scalar>
class CpTensor {
public:
    /// Default constructor
    CpTensor() = default;

    /// Construct a CP tensor from its factors
    CpTensor(const std::vector<Array<Scalar>> &factor);

    /// Default destructor
    ~CpTensor() = default;

    /// Compute and return entry of CP tensor at specified index
    Scalar operator()(const std::vector<std::size_t> &index) const;

    /// Write to file
    void WriteToFile(const std::string &file_name) const;

    /// Read from file
    ///
    /// @warning Destroys any exisitng data
    void ReadFromFile(const std::string &file_name);

private:
    std::size_t ndim_;
    std::vector<std::size_t> size_;
    std::size_t rank_;
    std::vector<Array<Scalar>> factor_;
};

}  // namespace TensorFact

#endif
