#include "tensorfact/tt_svd.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "utils.hpp"

tensorfact::TtTensor tensorfact::TtSvd(const std::vector<long> &size,
                                       const std::vector<double> &array,
                                       double relative_tolerance) {
    if (relative_tolerance <= std::numeric_limits<double>::epsilon()) {
        throw std::logic_error("Required accuracy is too small");
    }

    long ndim = size.size();
    std::vector<long> rank(ndim + 1);

    const double delta = relative_tolerance / std::sqrt(ndim - 1);

    // compute cores
    rank[0] = 1;

    std::vector<std::vector<double>> core(ndim);
    core[ndim - 1] = array;

    for (long d = 0; d < ndim - 1; ++d) {
        const long m = rank[d] * size[d];
        const long n = core[ndim - 1].size() / m;

        long r;
        std::vector<double> s;
        std::vector<double> Vt;
        TruncatedSvd(m, n, core[ndim - 1], delta, true, r, core[d], s, Vt);

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
    std::vector<double> param(offset[ndim]);
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
    return tensorfact::TtTensor(ndim, size, rank, param);
}
