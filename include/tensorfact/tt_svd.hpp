/// @file tt_svd.hpp

#ifndef TENSORFACT_TTSVD_HPP
#define TENSORFACT_TTSVD_HPP

#include <vector>

#include "tensorfact/tt_tensor.hpp"

namespace tensorfact {

/// Compute TT factorization from full tensors using TT-SVD
TtTensor TtSvd(const std::vector<long> &size, const std::vector<double> &array,
               double relative_tolerance);

}  // namespace tensorfact

#endif
