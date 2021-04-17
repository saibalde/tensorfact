/// @file cp_tensor.hpp

#ifndef TENSORFACT_CPTENSOR_HPP
#define TENSORFACT_CPTENSOR_HPP

#include <memory>
#include <string>
#include <vector>

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
class CpTensor {
public:
    /// Default constructor
    CpTensor() = default;

    /// Construct a CP tensor from its parameters
    CpTensor(int ndim, const std::vector<int> &size, int rank,
             const std::vector<double> &param);

    /// Default destructor
    ~CpTensor() = default;

    /// Number of parameters
    int NumParam() const { return offset_[ndim_]; }

    /// Compute and return entry of CP tensor at specified index
    double Entry(const std::vector<int> &index) const;

    /// Write to file
    void WriteToFile(const std::string &file_name) const;

    /// Read from file
    ///
    /// @warning Destroys any exisitng data
    void ReadFromFile(const std::string &file_name);

private:
    int LinearIndex(int i, int r, int d) const;

    int ndim_;
    std::vector<int> size_;
    int rank_;
    std::vector<int> offset_;
    std::vector<double> param_;
};

}  // namespace tensorfact

#endif
