#include "tensorfact/tt_svd.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "truncated_svd.hpp"

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtSvd(const std::vector<long> &size,
                                             const std::vector<Real> &array,
                                             Real relative_tolerance) {
    if (relative_tolerance <= std::numeric_limits<Real>::epsilon()) {
        throw std::logic_error("Required accuracy is too small");
    }

    long ndim = size.size();
    std::vector<long> rank(ndim + 1);

    const Real delta =
        relative_tolerance / std::sqrt(static_cast<Real>(ndim - 1));

    // compute cores
    rank[0] = 1;

    std::vector<std::vector<Real>> core(ndim);
    core[ndim - 1] = array;

    for (long d = 0; d < ndim - 1; ++d) {
        const long m = rank[d] * size[d];
        const long n = core[ndim - 1].size() / m;

        long r;
        std::vector<Real> s;
        std::vector<Real> Vt;
        TruncatedSvd<Real>(m, n, core[ndim - 1], delta, true, r, core[d], s,
                           Vt);

        core[ndim - 1].resize(r * n);
        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < r; ++i) {
                core[ndim - 1][i + j * r] = s[i] * Vt[i + j * r];
            }
        }

        rank[d + 1] = r;
    }

    rank[ndim] = 1;

    // calculate offsets
    std::vector<long> offset(ndim + 1);
    offset[0] = 0;
    for (long d = 0; d < ndim; ++d) {
        offset[d + 1] = offset[d] + rank[d] * size[d] * rank[d + 1];
    }

    // combine cores
    std::vector<Real> param(offset[ndim]);
    for (long d = 0; d < ndim; ++d) {
        for (long k = 0; k < rank[d + 1]; ++k) {
            for (long j = 0; j < size[d]; ++j) {
                for (long i = 0; i < rank[d]; ++i) {
                    param[i + j * rank[d] + k * rank[d] * size[d] + offset[d]] =
                        core[d][i + j * rank[d] + k * rank[d] * size[d]];
                }
            }
        }
    }

    // create TT tensor object
    return tensorfact::TtTensor<Real>(ndim, size, rank, param);
}

// explicit instantiations

template tensorfact::TtTensor<float> tensorfact::TtSvd(
    const std::vector<long> &size, const std::vector<float> &array,
    float relative_tolerance);
template tensorfact::TtTensor<double> tensorfact::TtSvd(
    const std::vector<long> &size, const std::vector<double> &array,
    double relative_tolerance);
