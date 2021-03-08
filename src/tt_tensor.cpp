#include "tensorfact/tt_tensor.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "svd.hpp"

template <typename Real>
tensorfact::TtTensor<Real>::TtTensor(const arma::field<arma::Cube<Real>> &cores)
    : core_(cores) {
    ndim_ = cores.n_elem;

    if ((cores[0].n_rows != 1) || (cores[ndim_ - 1].n_slices != 1)) {
        throw std::logic_error(
            "tensorfact::TtTensor::TtTensor() - Dimensionality of the cores is "
            "not compatible with TT format");
    }

    size_.zeros(ndim_);
    rank_.zeros(ndim_ + 1);

    for (arma::uword d = 0; d < ndim_; ++d) {
        size_(d) = cores(d).n_cols;

        rank_(d) = cores(d).n_rows;
        if ((d > 0) && (cores(d - 1).n_slices != rank_(d))) {
            throw std::logic_error(
                "tensorfact::TtTensor::TtTensor() - Dimensionality of the "
                "cores is not compatible with TT format");
        }
    }
    rank_[ndim_] = 1;

    core_ = cores;
}

template <typename Real>
Real tensorfact::TtTensor<Real>::operator()(
    const arma::Col<arma::uword> &index) const {
    arma::Mat<Real> temp;

    for (arma::uword d = 0; d < ndim_; ++d) {
        arma::Mat<Real> slice(rank_(d), rank_(d + 1));
        for (arma::uword j = 0; j < rank_(d + 1); ++j) {
            for (arma::uword i = 0; i < rank_(d); ++i) {
                slice(i, j) = core_(d)(i, index(d), j);
            }
        }

        if (d == 0) {
            temp = slice;
        } else {
            temp = temp * slice;
        }
    }

    return temp(0, 0);
}

template <typename Real>
void tensorfact::TtTensor<Real>::WriteToFile(
    const std::string &file_name) const {
    std::ofstream file(file_name);

    file << "TT Tensor" << std::endl;

    file << ndim_ << std::endl;

    for (arma::uword d = 0; d < ndim_; ++d) {
        file << size_(d) << std::endl;
    }

    for (arma::uword d = 0; d <= ndim_; ++d) {
        file << rank_(d) << std::endl;
    }

    file << std::scientific;

    for (arma::uword d = 0; d < ndim_; ++d) {
        for (arma::uword k = 0; k < rank_(d + 1); ++k) {
            for (arma::uword j = 0; j < size_(d); ++j) {
                for (arma::uword i = 0; i < rank_(d); ++i) {
                    file << std::setprecision(17) << core_(d)(i, j, k)
                         << std::endl;
                }
            }
        }
    }
}

template <typename Real>
void tensorfact::TtTensor<Real>::ReadFromFile(const std::string &file_name) {
    std::ifstream file(file_name);

    {
        std::string line;
        std::getline(file, line);
        if (line.compare("TT Tensor") != 0) {
            throw std::runtime_error(
                "tensorfact::TtTensor::ReadFromFile() - File does not seem to "
                "contain a TT Tensor");
        }
    }

    {
        std::string line;
        std::getline(file, line);
        std::istringstream line_stream(line);
        line_stream >> ndim_;
    }

    size_.set_size(ndim_);
    for (arma::uword d = 0; d < ndim_; ++d) {
        std::string line;
        std::getline(file, line);
        std::istringstream line_stream(line);
        line_stream >> size_(d);
    }

    rank_.set_size(ndim_ + 1);
    for (arma::uword d = 0; d <= ndim_; ++d) {
        std::string line;
        std::getline(file, line);
        std::istringstream line_stream(line);
        line_stream >> rank_(d);
    }

    core_.set_size(ndim_);
    for (arma::uword d = 0; d < ndim_; ++d) {
        core_(d).set_size(rank_(d), size_(d), rank_(d + 1));

        for (arma::uword k = 0; k < rank_(d + 1); ++k) {
            for (arma::uword j = 0; j < size_(d); ++j) {
                for (arma::uword i = 0; i < rank_(d); ++i) {
                    std::string line;
                    std::getline(file, line);
                    std::istringstream line_stream(line);
                    line_stream >> core_(d)(i, j, k);
                }
            }
        }
    }
}

template <typename Real>
void tensorfact::TtTensor<Real>::ComputeFromFull(
    const arma::Col<Real> &array, const arma::Col<arma::uword> &size,
    Real rel_acc) {
    if (!arma::all(size > 0)) {
        throw std::logic_error(
            "tensorfact::TtTensor::TtTensor() - Entries of the size tensor "
            "must be positive");
    }

    if (array.n_elem != arma::prod(size)) {
        throw std::logic_error(
            "tensorfact::TtTensor::TtTensor() - Number of array elements and "
            "array size does not match");
    }

    if (rel_acc < std::numeric_limits<Real>::epsilon()) {
        throw std::logic_error(
            "tensorfact::TtTensor::TtTensor() - Relative accuracy is too "
            "small");
    }

    ndim_ = size.n_elem;
    size_ = size;
    rank_ = arma::Col<arma::uword>(ndim_ + 1);
    core_ = arma::field<arma::Cube<Real>>(ndim_);

    const Real delta_squared =
        std::pow(rel_acc, 2) * arma::dot(array, array) / (ndim_ - 1);

    arma::Col<Real> array_copy(array);
    rank_(0) = 1;

    for (arma::uword d = 0; d < ndim_; ++d) {
        if (d < ndim_ - 1) {
            arma::Mat<Real> C(array_copy.memptr(), rank_(d) * size_(d),
                              array_copy.n_elem / (rank_(d) * size_(d)), false,
                              true);

            arma::Mat<Real> U;
            arma::Col<Real> s;
            arma::Mat<Real> V;
            tensorfact::TruncatedSvd<Real>(C, delta_squared, U, s, V,
                                           rank_(d + 1));

            core_(d) =
                arma::Cube<Real>(U.memptr(), rank_(d), size_(d), rank_(d + 1));

            array_copy = arma::vectorise(arma::diagmat(s) * V.t());
        } else {
            rank_(d + 1) = 1;
            core_(d) = arma::Cube<Real>(array_copy.memptr(), rank_(d), size_(d),
                                        rank_(d + 1));
        }
    }
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::Round(
    Real rel_acc) const {
    if (rel_acc < std::numeric_limits<Real>::epsilon()) {
        return *this;
    }

    // truncation parameter
    const Real delta_squared =
        std::pow(rel_acc, 2.0) * this->Dot(*this) / (ndim_ - 1);

    // make a copy of the cores and ranks
    arma::field<arma::Cube<Real>> core(ndim_);
    for (arma::uword d = 0; d < ndim_; ++d) {
        core(d) = core_(d);
    }

    arma::Col<arma::uword> rank(ndim_ + 1);
    for (arma::uword d = 0; d <= ndim_; ++d) {
        rank(d) = rank_(d);
    }

    // right-to-left orthogonalization
    for (arma::uword d = ndim_ - 1; d > 0; --d) {
        arma::Mat<Real> M1(core(d).memptr(), rank(d), size_(d) * rank(d + 1),
                           false, true);
        arma::Mat<Real> M2(core(d - 1).memptr(), rank(d - 1) * size_(d - 1),
                           rank(d), false, true);

        arma::Mat<Real> Q;
        arma::Mat<Real> R;
        arma::qr_econ(Q, R, M1.t());
        arma::uword r = Q.n_cols;

        {
            arma::Mat<Real> temp = Q.t();
            core(d) = arma::Cube<Real>(temp.memptr(), r, size_(d), rank(d + 1));
        }

        {
            arma::Mat<Real> temp = M2 * R.t();
            core(d - 1) =
                arma::Cube<Real>(temp.memptr(), rank(d - 1), size_(d - 1), r);
        }

        rank(d) = r;
    }

    // left-to-right compression
    for (arma::uword d = 0; d < ndim_ - 1; ++d) {
        arma::Mat<Real> M1(core(d).memptr(), rank(d) * size_(d), rank(d + 1),
                           false, true);
        arma::Mat<Real> M2(core(d + 1).memptr(), rank(d + 1),
                           size_(d + 1) * rank(d + 2), false, true);

        arma::Mat<Real> U;
        arma::Col<Real> s;
        arma::Mat<Real> V;
        arma::uword r;
        tensorfact::TruncatedSvd<Real>(M1, delta_squared, U, s, V, r);

        core(d) = arma::Cube<Real>(U.memptr(), rank(d), size_(d), r);

        {
            arma::Mat<Real> temp = arma::diagmat(s) * V.t() * M2;
            core(d + 1) =
                arma::Cube<Real>(temp.memptr(), r, size_(d + 1), rank(d + 2));
        }

        rank(d + 1) = r;
    }

    return tensorfact::TtTensor<Real>(core);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator+(
    const TtTensor<Real> &other) const {
    if (!arma::all(size_ == other.size_)) {
        throw std::logic_error(
            "tensorfact::TtTensor::operator+() - Sizes of the TT vectors does "
            "not match");
    }

    // new ranks
    arma::Col<arma::uword> rank_new = rank_ + other.rank_;
    rank_new(0) = 1;
    rank_new(ndim_) = 1;

    // new cores
    arma::field<arma::Cube<Real>> core_new(ndim_);

    // first core
    core_new(0).zeros(1, size_(0), rank_new(1));

    for (arma::uword k = 0; k < rank_(1); ++k) {
        for (arma::uword j = 0; j < size_(0); ++j) {
            core_new(0)(0, j, k) = core_(0)(0, j, k);
        }
    }

    for (arma::uword k = 0; k < other.rank_(1); ++k) {
        for (arma::uword j = 0; j < size_(0); ++j) {
            core_new(0)(0, j, rank_(1) + k) = other.core_(0)(0, j, k);
        }
    }

    // middle cores
    for (arma::uword d = 1; d < ndim_ - 1; ++d) {
        core_new(d).zeros(rank_new(d), size_(d), rank_new(d + 1));

        for (arma::uword k = 0; k < rank_(d + 1); ++k) {
            for (arma::uword j = 0; j < size_(d); ++j) {
                for (arma::uword i = 0; i < rank_(d); ++i) {
                    core_new(d)(i, j, k) = core_(d)(i, j, k);
                }
            }
        }

        for (arma::uword k = 0; k < other.rank_(d + 1); ++k) {
            for (arma::uword j = 0; j < size_(d); ++j) {
                for (arma::uword i = 0; i < other.rank_(d); ++i) {
                    core_new(d)(rank_(d) + i, j, rank_(d + 1) + k) =
                        other.core_(d)(i, j, k);
                }
            }
        }
    }

    // last core
    core_new(ndim_ - 1).zeros(rank_new(ndim_ - 1), size_(ndim_ - 1), 1);

    for (arma::uword j = 0; j < size_(ndim_ - 1); ++j) {
        for (arma::uword i = 0; i < rank_(ndim_ - 1); ++i) {
            core_new(ndim_ - 1)(i, j, 0) = core_(ndim_ - 1)(i, j, 0);
        }

        for (arma::uword i = 0; i < other.rank_(ndim_ - 1); ++i) {
            core_new(ndim_ - 1)(rank_(ndim_ - 1) + i, j, 0) =
                other.core_(ndim_ - 1)(i, j, 0);
        }
    }

    return TtTensor<Real>(core_new);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator-(
    const tensorfact::TtTensor<Real> &other) const {
    return *this + other * (-1.0);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator*(
    Real alpha) const {
    arma::field<arma::Cube<Real>> core_new(ndim_);

    for (arma::uword d = 0; d < ndim_; ++d) {
        core_new(d) = core_(d);
    }

    core_new(0) *= alpha;

    return TtTensor<Real>(core_new);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator/(
    Real alpha) const {
    if (std::abs(alpha) < std::numeric_limits<Real>::epsilon()) {
        throw std::logic_error(
            "tensorfact::TtTensor::operator/() - Scalar is too close to zero");
    }

    return *this * (1 / alpha);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::Concatenate(
    const tensorfact::TtTensor<Real> &other, arma::uword dim,
    Real rel_acc) const {
    if (ndim_ != other.ndim_) {
        throw std::invalid_argument(
            "tensorfact::TtTensor::Concatenate() - The dimensionality of the "
            "two tensor must match");
    }

    if (dim >= ndim_) {
        throw std::invalid_argument(
            "tensorfact::TtTensor::Concatenate() - Cannot concatenate along "
            "specified dimension");
    }

    for (arma::uword d = 0; d < ndim_; ++d) {
        if (d != dim && size_(d) != other.size_(d)) {
            throw std::invalid_argument(
                "tensorfact::TtTensor::Concatenate() - Tensor sizes must match "
                "except along the concatenation dimension");
        }
    }

    const arma::uword size_1 = size_(dim);
    const arma::uword size_2 = other.size_(dim);

    const tensorfact::TtTensor<Real> tensor_1 =
        this->AddZeroPaddingBack(dim, size_2);
    const tensorfact::TtTensor<Real> tensor_2 =
        other.AddZeroPaddingFront(dim, size_1);

    return (tensor_1 + tensor_2).Round(rel_acc);
}

template <typename Real>
Real tensorfact::TtTensor<Real>::Dot(
    const tensorfact::TtTensor<Real> &other) const {
    if (arma::any(size_ != other.size_)) {
        throw std::logic_error(
            "tensorfact::TtTensor::Dot() - Sizes of the two TT vectors does "
            "not match");
    }

    arma::Cube<Real> temp_3d;
    arma::Mat<Real> temp_2d;

    for (arma::uword l = 0; l < ndim_; ++l) {
        arma::uword d = ndim_ - 1 - l;

        temp_3d.set_size(other.rank_(d), rank_(d), size_(d));

        if (d == ndim_ - 1) {
            // Kronecker product of the last cores
            for (arma::uword k = 0; k < size_(d); ++k) {
                arma::Mat<Real> slice(rank_(d), rank_(d + 1));
                for (arma::uword j = 0; j < rank_(d + 1); ++j) {
                    for (arma::uword i = 0; i < rank_(d); ++i) {
                        slice(i, j) = core_(d)(i, k, j);
                    }
                }

                arma::Mat<Real> other_slice(other.rank_(d), other.rank_(d + 1));
                for (arma::uword j = 0; j < other.rank_(d + 1); ++j) {
                    for (arma::uword i = 0; i < other.rank_(d); ++i) {
                        other_slice(i, j) = other.core_(d)(i, k, j);
                    }
                }

                temp_3d.slice(k) = other_slice * slice.t();
            }
        } else {
            // multiplication by Kronecker product of the cores
            for (arma::uword k = 0; k < size_(d); ++k) {
                arma::Mat<Real> slice(rank_(d), rank_(d + 1));
                for (arma::uword j = 0; j < rank_(d + 1); ++j) {
                    for (arma::uword i = 0; i < rank_(d); ++i) {
                        slice(i, j) = core_(d)(i, k, j);
                    }
                }

                arma::Mat<Real> other_slice(other.rank_(d), other.rank_(d + 1));
                for (arma::uword j = 0; j < other.rank_(d + 1); ++j) {
                    for (arma::uword i = 0; i < other.rank_(d); ++i) {
                        other_slice(i, j) = other.core_(d)(i, k, j);
                    }
                }

                temp_3d.slice(k) = other_slice * (temp_2d * slice.t());
            }
        }

        temp_2d = arma::sum(temp_3d, 2);
    }

    return temp_2d(0, 0);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::AddZeroPaddingBack(
    arma::uword dim, arma::uword pad) const {
    arma::field<arma::Cube<Real>> core(ndim_);

    for (arma::uword d = 0; d < ndim_; ++d) {
        if (d != dim) {
            core(d) = core_(d);
        }
    }

    core(dim).set_size(rank_(dim), size_(dim) + pad, rank_(dim + 1));
    for (arma::uword k = 0; k < rank_(dim + 1); ++k) {
        for (arma::uword j = 0; j < size_(dim); ++j) {
            for (arma::uword i = 0; i < rank_(dim); ++i) {
                core(dim)(i, j, k) = core_(dim)(i, j, k);
            }
        }

        for (arma::uword j = 0; j < pad; ++j) {
            for (arma::uword i = 0; i < rank_(dim); ++i) {
                core(dim)(i, j + size_(dim), k) = static_cast<Real>(0);
            }
        }
    }

    return tensorfact::TtTensor<Real>(core);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::AddZeroPaddingFront(
    arma::uword dim, arma::uword pad) const {
    arma::field<arma::Cube<Real>> core(ndim_);

    for (arma::uword d = 0; d < ndim_; ++d) {
        if (d != dim) {
            core(d) = core_(d);
        }
    }

    core(dim).set_size(rank_(dim), size_(dim) + pad, rank_(dim + 1));
    for (arma::uword k = 0; k < rank_(dim + 1); ++k) {
        for (arma::uword j = 0; j < pad; ++j) {
            for (arma::uword i = 0; i < rank_(dim); ++i) {
                core(dim)(i, j, k) = static_cast<Real>(0);
            }
        }

        for (arma::uword j = 0; j < size_(dim); ++j) {
            for (arma::uword i = 0; i < rank_(dim); ++i) {
                core(dim)(i, j + pad, k) = core_(dim)(i, j, k);
            }
        }
    }

    return tensorfact::TtTensor<Real>(core);
}

// explicit instantiations -----------------------------------------------------

template class tensorfact::TtTensor<float>;
template class tensorfact::TtTensor<double>;
