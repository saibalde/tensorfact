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

    // make copy of array
    std::vector<Real> array_copy(array);

    // compute cores
    rank[0] = 1;
    std::vector<arma::Cube<Real>> core(ndim);

    for (long d = 0; d < ndim - 1; ++d) {
        arma::Mat<Real> A(array_copy.data(), rank[d] * size[d], array_copy.size() / (rank[d] * size[d]));

        arma::Mat<Real> U;
        arma::Col<Real> s;
        arma::Mat<Real> V;
        long r;
        TruncatedSvd<Real>(A, delta, true, U, s, V, r);

        core[d] = arma::Cube<Real>(U.memptr(), rank[d], size[d], r);

        arma::Mat<Real> temp = arma::diagmat(s) * V.t();
        array_copy = std::vector<Real>(temp.memptr(), temp.memptr() + temp.n_elem);

        rank[d + 1] = r;
    }

    core[ndim - 1] = arma::Cube<Real>(array_copy.data(), rank[ndim - 1], size[ndim - 1], 1);
    rank[ndim] = 1;

    // construct TT tensor
    tensorfact::TtTensor<Real> tt_tensor(ndim, size, rank);
    for (long d = 0; d < ndim; ++d) {
        for (long k = 0; k < rank[d + 1]; ++k) {
            for (long j = 0; j < size[d]; ++j) {
                for (long i = 0; i < rank[d]; ++i) {
                    tt_tensor.Param(i, j, k, d) = core[d](i, j, k);
                }
            }
        }
    }

    return tt_tensor;
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

    // make copy of array
    std::vector<Real> array_copy(array);

    // compute cores
    rank[0] = 1;
    std::vector<arma::Cube<Real>> core(ndim);

    for (long d = 0; d < ndim - 1; ++d) {
        arma::Mat<Real> A(array_copy.data(), rank[d] * size[d], array_copy.size() / (rank[d] * size[d]));

        arma::Mat<Real> U_temp;
        arma::Col<Real> s_temp;
        arma::Mat<Real> V_temp;
        long r_temp;
        TruncatedSvd<Real>(A, 0.0, false, U_temp, s_temp, V_temp, r_temp);

        long r = std::min(r_temp, max_rank);
        arma::Mat<Real> U(U_temp.memptr(), U_temp.n_rows, r);
        arma::Col<Real> s(s_temp.memptr(), r);
        arma::Mat<Real> V(V_temp.memptr(), V_temp.n_rows, r);

        core[d] = arma::Cube<Real>(U.memptr(), rank[d], size[d], r);

        arma::Mat<Real> temp = arma::diagmat(s) * V.t();
        array_copy = std::vector<Real>(temp.memptr(), temp.memptr() + temp.n_elem);

        rank[d + 1] = r;
    }

    core[ndim - 1] = arma::Cube<Real>(array_copy.data(), rank[ndim - 1], size[ndim - 1], 1);
    rank[ndim] = 1;

    // construct TT tensor
    tensorfact::TtTensor<Real> tt_tensor(ndim, size, rank);
    for (long d = 0; d < ndim; ++d) {
        for (long k = 0; k < rank[d + 1]; ++k) {
            for (long j = 0; j < size[d]; ++j) {
                for (long i = 0; i < rank[d]; ++i) {
                    tt_tensor.Param(i, j, k, d) = core[d](i, j, k);
                }
            }
        }
    }

    return tt_tensor;
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
