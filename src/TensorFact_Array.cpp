#include "TensorFact_Array.hpp"

#include <limits>
#include <stdexcept>

#include "blas.hh"
#include "lapack.hh"

template <typename Scalar>
const Scalar &TensorFact::Array<Scalar>::operator()(
    const std::vector<std::size_t> &cartesian_index) const {
    std::size_t linear_index;
    CartesianToLinearIndex(cartesian_index, linear_index);
    return entries_[linear_index];
}

template <typename Scalar>
Scalar &TensorFact::Array<Scalar>::operator()(
    const std::vector<std::size_t> &cartesian_index) {
    std::size_t linear_index;
    CartesianToLinearIndex(cartesian_index, linear_index);
    return entries_[linear_index];
}

template <typename Scalar>
void TensorFact::Array<Scalar>::Reshape(
    const std::vector<std::size_t> &size_new) {
    const std::size_t ndim_new = size_new.size();

    std::vector<std::size_t> unfolding_factors_new(ndim_new + 1);
    unfolding_factors_new[0] = 1;
    for (std::size_t d = 0; d < ndim_new; ++d) {
        unfolding_factors_new[d + 1] = unfolding_factors_new[d] * size_new[d];
    }

    if (unfolding_factors_[ndim_] != unfolding_factors_new[ndim_new]) {
        throw std::logic_error(
            "Total number of elements should be unchanged during reshape");
    }

    ndim_ = ndim_new;
    size_ = size_new;
    unfolding_factors_ = unfolding_factors_new;
}

template <typename Scalar>
void TensorFact::Array<Scalar>::Resize(const std::vector<std::size_t> &size) {
    ndim_ = size.size();

    size_ = size;

    unfolding_factors_.resize(ndim_ + 1);
    if (ndim_ == 0) {
        unfolding_factors_[0] = 0;
    } else {
        unfolding_factors_[0] = 1;
        for (std::size_t d = 0; d < ndim_; ++d) {
            unfolding_factors_[d + 1] = unfolding_factors_[d] * size_[d];
        }
    }

    entries_.resize(unfolding_factors_[ndim_]);
}

template <typename Scalar>
Scalar TensorFact::Array<Scalar>::FrobeniusNorm() const {
    return blas::nrm2(unfolding_factors_[ndim_], entries_.data(), 1);
}

template <typename Scalar>
void TensorFact::Array<Scalar>::Multiply(
    bool conjugate, const TensorFact::Array<Scalar> &other,
    bool other_conjugate, TensorFact::Array<Scalar> &result) const {
    if (ndim_ == 2 && other.ndim_ == 1) {
        if (!conjugate) {
            if (size_[1] != other.size_[0]) {
                throw std::logic_error(
                    "Dimension mismatch in matrix-vector product");
            }

            result.Resize({size_[0]});

            blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, size_[0],
                       size_[1], static_cast<Scalar>(1), entries_.data(),
                       size_[0], other.entries_.data(), 1,
                       static_cast<Scalar>(0), result.entries_.data(), 1);
        } else {
            if (size_[0] != other.size_[0]) {
                throw std::logic_error(
                    "Dimension mismatch in matrix-vector product");
            }

            result.Resize({size_[1]});

            blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, size_[0],
                       size_[1], static_cast<Scalar>(1), entries_.data(),
                       size_[0], other.entries_.data(), 1,
                       static_cast<Scalar>(0), result.entries_.data(), 1);
        }
    } else if (ndim_ == 2 && other.ndim_ == 2) {
        if (!conjugate && !other_conjugate) {
            if (size_[1] != other.size_[0]) {
                throw std::logic_error(
                    "Dimension mismatch in matrix-matrix product");
            }

            result.Resize({size_[0], other.size_[1]});

            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                       blas::Op::NoTrans, size_[0], other.size_[1], size_[1],
                       static_cast<Scalar>(1), entries_.data(), size_[0],
                       other.entries_.data(), size_[1], static_cast<Scalar>(0),
                       result.entries_.data(), size_[0]);
        } else if (!conjugate && other_conjugate) {
            if (size_[1] != other.size_[1]) {
                throw std::logic_error(
                    "Dimension mismatch in matrix-matrix product");
            }

            result.Resize({size_[0], other.size_[0]});

            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                       blas::Op::ConjTrans, size_[0], other.size_[0], size_[1],
                       static_cast<Scalar>(1), entries_.data(), size_[0],
                       other.entries_.data(), other.size_[0],
                       static_cast<Scalar>(0), result.entries_.data(),
                       size_[0]);
        } else if (conjugate && !other_conjugate) {
            if (size_[0] != other.size_[0]) {
                throw std::logic_error(
                    "Dimension mismatch in matrix-matrix product");
            }

            result.Resize({size_[1], other.size_[1]});

            blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans,
                       blas::Op::NoTrans, size_[1], other.size_[1], size_[0],
                       static_cast<Scalar>(1), entries_.data(), size_[0],
                       other.entries_.data(), other.size_[0],
                       static_cast<Scalar>(0), result.entries_.data(),
                       size_[1]);
        } else {
            if (size_[0] != other.size_[1]) {
                throw std::logic_error(
                    "Dimension mismatch in matrix-matrix product");
            }

            result.Resize({size_[1], other.size_[0]});

            blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans,
                       blas::Op::ConjTrans, size_[1], other.size_[0], size_[0],
                       static_cast<Scalar>(1), entries_.data(), size_[0],
                       other.entries_.data(), other.size_[0],
                       static_cast<Scalar>(0), result.entries_.data(),
                       size_[1]);
        }
    } else {
        throw std::invalid_argument(
            "Only matrix-matrix and matrix-vector products are implemented");
    }
}

template <typename Scalar>
void TensorFact::Array<Scalar>::TruncatedSvd(TensorFact::Array<Scalar> &U,
                                             TensorFact::Array<Scalar> &s,
                                             TensorFact::Array<Scalar> &Vt,
                                             Scalar tolerance,
                                             bool relative_flag) const {
    if (ndim_ != 2) {
        throw std::invalid_argument("Only matrix SVD is implemented");
    }

    TensorFact::Array<Scalar> U_thin;
    TensorFact::Array<Scalar> s_thin;
    TensorFact::Array<Scalar> Vt_thin;

    const std::size_t k = std::min(size_[0], size_[1]);

    U_thin.Resize({size_[0], k});
    s_thin.Resize({k});
    Vt_thin.Resize({k, size_[1]});

    std::vector<Scalar> entries = entries_;

    const std::size_t status =
        lapack::gesdd(lapack::Job::SomeVec, size_[0], size_[1], entries.data(),
                      size_[0], s_thin.entries_.data(), U_thin.entries_.data(),
                      size_[0], Vt_thin.entries_.data(), k);
    if (status != 0) {
        throw std::runtime_error("SVD computation did not converge");
    }

    std::size_t rank = k;
    if (tolerance <= std::numeric_limits<Scalar>::epsilon()) {
        while ((rank > 0) &&
               (s_thin(rank - 1) <= std::numeric_limits<Scalar>::epsilon())) {
            --rank;
        }
    } else {
        Scalar max_frobenius_error;
        if (!relative_flag) {
            max_frobenius_error = std::pow(tolerance, 2);
        } else {
            max_frobenius_error =
                std::pow(tolerance * s_thin.FrobeniusNorm(), 2);
        }

        Scalar frobenius_error = static_cast<Scalar>(0);
        while ((rank > 0) && (frobenius_error <= max_frobenius_error)) {
            frobenius_error += std::pow(s_thin({rank - 1}), 2);
            --rank;
        }
    }

    U.Resize({size_[0], rank});
    for (std::size_t r = 0; r < rank; ++r) {
        for (std::size_t i = 0; i < size_[0]; ++i) {
            U({i, r}) = U_thin({i, r});
        }
    }

    s.Resize({rank});
    for (std::size_t r = 0; r < rank; ++r) {
        s({r}) = s_thin({r});
    }

    Vt.Resize({rank, size_[1]});
    for (std::size_t j = 0; j < size_[1]; ++j) {
        for (std::size_t r = 0; r < rank; ++r) {
            Vt({r, j}) = Vt_thin({r, j});
        }
    }
}

template <typename Scalar>
void TensorFact::Array<Scalar>::CartesianToLinearIndex(
    const std::vector<std::size_t> &cartesian_index,
    std::size_t &linear_index) const {
    linear_index = 0;
    for (std::size_t d = 0; d < ndim_; ++d) {
        linear_index += cartesian_index[d] * unfolding_factors_[d];
    }
}

template <typename Scalar>
void TensorFact::Array<Scalar>::LinearToCartesianIndex(
    std::size_t linear_index, std::vector<std::size_t> &cartesian_index) const {
    cartesian_index.resize(ndim_);
    for (std::size_t d = 0; d < ndim_; ++d) {
        cartesian_index[d] = linear_index % unfolding_factors_[d + 1];
        linear_index /= unfolding_factors_[d + 1];
    }
}

// explicit instantiations -----------------------------------------------------

template class TensorFact::Array<float>;
template class TensorFact::Array<double>;
