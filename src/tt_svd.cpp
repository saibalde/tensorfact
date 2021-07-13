#include "tensorfact/tt_svd.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "truncated_svd.hpp"

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtSvd(const std::vector<long> &size,
                                             const std::vector<Real> &array,
                                             Real relative_tolerance) {
    if (relative_tolerance < std::numeric_limits<Real>::epsilon()) {
        throw std::logic_error("Specified relative tolerance is too small");
    }

    if (relative_tolerance >= 1) {
        throw std::logic_error("Relative tolerance must be smaller than 1");
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

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtSvd(const std::vector<long> &size,
                                             const std::vector<Real> &array,
                                             long max_rank) {
    if (max_rank < 1) {
        throw std::logic_error("Maximum rank must be at least 1");
    }

    long ndim = size.size();
    std::vector<long> rank(ndim + 1);

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
        TruncatedSvd<Real>(m, n, core[ndim - 1], 0.0, false, r, core[d], s, Vt);

        {
            const long r_new = std::min(r, max_rank);

            std::vector<Real> U_new(m * r_new);
            for (long j = 0; j < r_new; ++j) {
                for (long i = 0; i < m; ++i) {
                    U_new[i + j * m] = core[d][i + j * m];
                }
            }
            core[d] = std::move(U_new);

            std::vector<Real> s_new(r_new);
            for (long i = 0; i < r_new; ++i) {
                s_new[i] = s[i];
            }
            s = std::move(s_new);

            std::vector<Real> Vt_new(r_new * n);
            for (long j = 0; j < n; ++j) {
                for (long i = 0; i < r_new; ++i) {
                    Vt_new[i + j * r_new] = Vt[i + j * r];
                }
            }
            Vt = std::move(Vt_new);

            r = r_new;
        }

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

template tensorfact::TtTensor<float> tensorfact::TtSvd(
    const std::vector<long> &size, const std::vector<float> &array,
    long max_rank);
template tensorfact::TtTensor<double> tensorfact::TtSvd(
    const std::vector<long> &size, const std::vector<double> &array,
    long max_rank);
