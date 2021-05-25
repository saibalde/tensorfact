#include "tensorfact/tt_tensor.hpp"

#include <hdf5.h>

#include <blas.hh>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "utils.hpp"

template <typename Real>
tensorfact::TtTensor<Real>::TtTensor(long ndim, const std::vector<long> &size,
                                     const std::vector<long> &rank,
                                     const std::vector<Real> &param)
    : ndim_(ndim), size_(size), rank_(rank), offset_(ndim + 1), param_(param) {
    if (ndim_ < 1) {
        throw std::invalid_argument("Dimension must be positive");
    }

    if (size_.size() != ndim_) {
        throw std::invalid_argument(
            "Length of the size vector is incompatible with dimension");
    }
    for (long d = 0; d < ndim_; ++d) {
        if (size_[d] < 1) {
            throw std::invalid_argument(
                "Entries of the size vector must be positive");
        }
    }

    if (rank_.size() != ndim_ + 1) {
        throw std::invalid_argument(
            "Length of the rank vector is incompatible with dimension");
    }
    if (rank_[0] != 1 || rank_[ndim_] != 1) {
        throw std::invalid_argument("Boundary ranks must be one");
    }
    for (long d = 1; d < ndim_; ++d) {
        if (rank_[d] < 1) {
            throw std::invalid_argument(
                "Entries of the rank vector must be positive");
        }
    }

    offset_[0] = 0;
    for (long d = 0; d < ndim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    if (param_.size() != offset_[ndim_]) {
        throw std::invalid_argument(
            "Length of the parameter vector is incompatible with dimension, "
            "size and rank");
    }
}

template <typename Real>
long tensorfact::TtTensor<Real>::NumElement() const {
    long num_element = 1;

    for (long d = 0; d < ndim_; ++d) {
        num_element *= size_[d];
    }

    return num_element;
}

template <typename Real>
Real tensorfact::TtTensor<Real>::Entry(const std::vector<long> &index) const {
    auto temp = std::make_shared<std::vector<Real>>();

    for (long d = ndim_ - 1; d >= 0; --d) {
        auto slice =
            std::make_shared<std::vector<Real>>(rank_[d] * rank_[d + 1]);

        for (long j = 0; j < rank_[d + 1]; ++j) {
            for (long i = 0; i < rank_[d]; ++i) {
                slice->at(i + j * rank_[d]) =
                    param_[LinearIndex(i, index[d], j, d)];
            }
        }

        if (d == ndim_ - 1) {
            std::swap(temp, slice);
        } else {
            auto temp_new = std::make_shared<std::vector<Real>>(rank_[d]);
            blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, rank_[d],
                       rank_[d + 1], static_cast<Real>(1), slice->data(),
                       rank_[d], temp->data(), 1, static_cast<Real>(0),
                       temp_new->data(), 1);
            std::swap(temp, temp_new);
        }
    }

    return temp->at(0);
}

template <typename Real>
void tensorfact::TtTensor<Real>::WriteToFile(
    const std::string &file_name) const {
    hid_t file =
        H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    {
        hsize_t dims[1] = {static_cast<hsize_t>(ndim_)};
        hid_t data_space = H5Screate_simple(1, dims, NULL);

        hid_t data_type = H5Tcopy(H5T_NATIVE_LONG);
        H5Tset_order(data_type, H5T_ORDER_LE);

        hid_t data_set =
            H5Dcreate(file, "tt_tensor_size", data_type, data_space,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(data_set, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 size_.data());

        H5Dclose(data_set);
        H5Tclose(data_type);
        H5Sclose(data_space);
    }

    {
        hsize_t dims[1] = {static_cast<hsize_t>(ndim_ + 1)};
        hid_t data_space = H5Screate_simple(1, dims, NULL);

        hid_t data_type = H5Tcopy(H5T_NATIVE_LONG);
        H5Tset_order(data_type, H5T_ORDER_LE);

        hid_t data_set =
            H5Dcreate(file, "tt_tensor_rank", data_type, data_space,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(data_set, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 rank_.data());

        H5Dclose(data_set);
        H5Tclose(data_type);
        H5Sclose(data_space);
    }

    {
        hsize_t dims[1] = {param_.size()};
        hid_t data_space = H5Screate_simple(1, dims, NULL);

        hid_t data_type;
        if (std::is_same<scalar_type, double>::value) {
            data_type = H5Tcopy(H5T_NATIVE_DOUBLE);
        } else if (std::is_same<scalar_type, float>::value) {
            data_type = H5Tcopy(H5T_NATIVE_FLOAT);
        }
        H5Tset_order(data_type, H5T_ORDER_LE);

        hid_t data_set =
            H5Dcreate(file, "tt_tensor_param", data_type, data_space,
                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (std::is_same<scalar_type, double>::value) {
            H5Dwrite(data_set, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     param_.data());
        } else if (std::is_same<scalar_type, float>::value) {
            H5Dwrite(data_set, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     param_.data());
        }

        H5Dclose(data_set);
        H5Tclose(data_type);
        H5Sclose(data_space);
    }

    H5Fclose(file);
}

template <typename Real>
void tensorfact::TtTensor<Real>::ReadFromFile(const std::string &file_name) {
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    {
        hid_t data_set = H5Dopen(file, "tt_tensor_size", H5P_DEFAULT);
        hid_t file_space = H5Dget_space(data_set);

        if (H5Sget_simple_extent_ndims(file_space) != 1) {
            throw std::runtime_error(
                "tt_tensor_size dataset must be one dimensional");
        }

        hsize_t dims[1];
        H5Sget_simple_extent_dims(file_space, dims, NULL);

        ndim_ = static_cast<long>(dims[0]);
        size_.resize(ndim_);

        hid_t mem_space = H5Screate_simple(1, dims, NULL);
        H5Dread(data_set, H5T_NATIVE_LONG, mem_space, file_space, H5P_DEFAULT,
                size_.data());

        H5Sclose(mem_space);
        H5Sclose(file_space);
        H5Dclose(data_set);
    }

    {
        hid_t data_set = H5Dopen(file, "tt_tensor_rank", H5P_DEFAULT);
        hid_t file_space = H5Dget_space(data_set);

        if (H5Sget_simple_extent_ndims(file_space) != 1) {
            throw std::runtime_error(
                "tt_tensor_rank dataset must be one dimensional");
        }

        hsize_t dims[1];
        H5Sget_simple_extent_dims(file_space, dims, NULL);

        if (static_cast<long>(dims[0]) != ndim_ + 1) {
            throw std::runtime_error(
                "Mismatch in sizes of tt_tensor_size and tt_tensor_rank");
        }

        rank_.resize(ndim_ + 1);

        hid_t mem_space = H5Screate_simple(1, dims, NULL);
        H5Dread(data_set, H5T_NATIVE_LONG, mem_space, file_space, H5P_DEFAULT,
                rank_.data());

        H5Sclose(mem_space);
        H5Sclose(file_space);
        H5Dclose(data_set);
    }

    offset_.resize(ndim_ + 1);
    offset_[0] = 0;
    for (long d = 0; d < ndim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    {
        hid_t data_set = H5Dopen(file, "tt_tensor_param", H5P_DEFAULT);
        hid_t file_space = H5Dget_space(data_set);

        if (H5Sget_simple_extent_ndims(file_space) != 1) {
            throw std::runtime_error(
                "tt_tensor_param dataset must be one dimensional");
        }

        hsize_t dims[1];
        H5Sget_simple_extent_dims(file_space, dims, NULL);

        if (static_cast<long>(dims[0]) != offset_[ndim_]) {
            throw std::runtime_error(
                "Mismatch in sizes of tt_tensor_size, tt_tensor_rank and "
                "tt_tensor_param");
        }

        param_.resize(offset_[ndim_]);

        hid_t mem_space = H5Screate_simple(1, dims, NULL);
        if (std::is_same<scalar_type, double>::value) {
            H5Dread(data_set, H5T_NATIVE_DOUBLE, mem_space, file_space,
                    H5P_DEFAULT, param_.data());
        } else if (std::is_same<scalar_type, float>::value) {
            H5Dread(data_set, H5T_NATIVE_FLOAT, mem_space, file_space,
                    H5P_DEFAULT, param_.data());
        }

        H5Sclose(mem_space);
        H5Sclose(file_space);
        H5Dclose(data_set);
    }

    H5Fclose(file);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator+(
    const TtTensor<Real> &other) const {
    if (ndim_ != other.ndim_) {
        throw std::invalid_argument("TT tensors must have same dimensionality");
    }

    for (long d = 0; d < ndim_; ++d) {
        if (size_[d] != other.size_[d]) {
            throw std::invalid_argument("TT tensors must have same size");
        }
    }

    // new ranks
    std::vector<long> rank_new(ndim_ + 1);
    rank_new[0] = 1;
    for (long d = 1; d < ndim_; ++d) {
        rank_new[d] = rank_[d] + other.rank_[d];
    }
    rank_new[ndim_] = 1;

    // new offsets
    std::vector<long> offset_new(ndim_ + 1);
    offset_new[0] = 0;
    for (long d = 0; d < ndim_; ++d) {
        offset_new[d + 1] =
            offset_new[d] + rank_new[d] * size_[d] * rank_new[d + 1];
    }

    // new parameters
    std::vector<Real> param_new(offset_new[ndim_], static_cast<Real>(0));

    // first core
    for (long k = 0; k < rank_[1]; ++k) {
        for (long j = 0; j < size_[0]; ++j) {
            param_new[j + k * size_[0] + offset_new[0]] =
                param_[LinearIndex(0, j, k, 0)];
        }
    }

    for (long k = 0; k < other.rank_[1]; ++k) {
        for (long j = 0; j < size_[0]; ++j) {
            param_new[j + (k + rank_[1]) * size_[0] + offset_new[0]] =
                other.param_[other.LinearIndex(0, j, k, 0)];
        }
    }

    // middle cores
    for (long d = 1; d < ndim_ - 1; ++d) {
        for (long k = 0; k < rank_[d + 1]; ++k) {
            for (long j = 0; j < size_[d]; ++j) {
                for (long i = 0; i < rank_[d]; ++i) {
                    param_new[i + j * rank_new[d] + k * rank_new[d] * size_[d] +
                              offset_new[d]] = param_[LinearIndex(i, j, k, d)];
                }
            }
        }

        for (long k = 0; k < other.rank_[d + 1]; ++k) {
            for (long j = 0; j < size_[d]; ++j) {
                for (long i = 0; i < other.rank_[d]; ++i) {
                    param_new[i + rank_[d] + j * rank_new[d] +
                              (k + rank_[d + 1]) * rank_new[d] * size_[d] +
                              offset_new[d]] =
                        other.param_[other.LinearIndex(i, j, k, d)];
                }
            }
        }
    }

    // last core
    for (long j = 0; j < size_[ndim_ - 1]; ++j) {
        for (long i = 0; i < rank_[ndim_ - 1]; ++i) {
            param_new[i + j * rank_new[ndim_ - 1] + offset_new[ndim_ - 1]] =
                param_[LinearIndex(i, j, 0, ndim_ - 1)];
        }

        for (long i = 0; i < other.rank_[ndim_ - 1]; ++i) {
            param_new[i + rank_[ndim_ - 1] + j * rank_new[ndim_ - 1] +
                      offset_new[ndim_ - 1]] =
                other.param_[other.LinearIndex(i, j, 0, ndim_ - 1)];
        }
    }

    return TtTensor(ndim_, size_, rank_new, param_new);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator*(
    Real alpha) const {
    std::vector<Real> param_new(param_);

    for (long n = 0; n < offset_[1]; ++n) {
        param_new[n] *= alpha;
    }

    return TtTensor(ndim_, size_, rank_, param_new);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator-(
    const tensorfact::TtTensor<Real> &other) const {
    return *this + other * static_cast<Real>(-1);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator/(
    Real alpha) const {
    if (std::abs(alpha) < std::numeric_limits<Real>::epsilon()) {
        throw std::logic_error("Dividing by a value too close to zero");
    }

    return *this * (static_cast<Real>(1) / alpha);
}

template <typename Real>
void tensorfact::TtTensor<Real>::Round(Real relative_tolerance) {
    // create cores
    std::vector<std::vector<Real>> core(ndim_);
    for (long d = 0; d < ndim_; ++d) {
        core[d].resize(rank_[d] * size_[d] * rank_[d + 1]);
        for (long k = 0; k < rank_[d + 1]; ++k) {
            for (long j = 0; j < size_[d]; ++j) {
                for (long i = 0; i < rank_[d]; ++i) {
                    core[d][i + j * rank_[d] + k * rank_[d] * size_[d]] =
                        param_[LinearIndex(i, j, k, d)];
                }
            }
        }
    }

    // right-to-left orthogonalization
    for (long d = ndim_ - 1; d > 0; --d) {
        const long m = rank_[d];
        const long n = size_[d] * rank_[d + 1];
        const long k = std::min(m, n);

        std::vector<Real> R;
        std::vector<Real> Q;
        ThinRq(m, n, core[d], R, Q);

        core[d] = Q;

        std::vector<Real> temp = core[d - 1];

        const long mm = rank_[d - 1] * size_[d - 1];
        core[d - 1].resize(mm * k);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   mm, k, m, static_cast<Real>(1), temp.data(), mm, R.data(), m,
                   static_cast<Real>(0), core[d - 1].data(), mm);

        rank_[d] = k;
    }

    // left-to-right compression
    if (relative_tolerance > std::numeric_limits<Real>::epsilon()) {
        const Real delta =
            relative_tolerance / std::sqrt(static_cast<Real>(ndim_ - 1));

        for (long d = 0; d < ndim_ - 1; ++d) {
            const long m = rank_[d] * size_[d];
            const long n = rank_[d + 1];

            std::vector<Real> U;
            std::vector<Real> s;
            std::vector<Real> Vt;
            long r;
            TruncatedSvd(m, n, core[d], delta, true, r, U, s, Vt);

            core[d] = U;

            for (long j = 0; j < n; ++j) {
                for (long i = 0; i < r; ++i) {
                    Vt[i + j * r] *= s[i];
                }
            }

            std::vector<Real> temp = core[d + 1];

            const long nn = size_[d + 1] * rank_[d + 2];
            core[d + 1].resize(r * nn);
            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                       blas::Op::NoTrans, r, nn, n, static_cast<Real>(1),
                       Vt.data(), r, temp.data(), n, static_cast<Real>(0),
                       core[d + 1].data(), r);

            rank_[d + 1] = r;
        }
    }

    // recalculate offsets
    offset_.resize(ndim_ + 1);
    offset_[0] = 0;
    for (long d = 0; d < ndim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    // combine cores
    param_.resize(offset_[ndim_]);
    for (long d = 0; d < ndim_; ++d) {
        for (long k = 0; k < rank_[d + 1]; ++k) {
            for (long j = 0; j < size_[d]; ++j) {
                for (long i = 0; i < rank_[d]; ++i) {
                    param_[LinearIndex(i, j, k, d)] =
                        core[d][i + j * rank_[d] + k * rank_[d] * size_[d]];
                }
            }
        }
    }
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::Concatenate(
    const tensorfact::TtTensor<Real> &other, long dim,
    Real relative_tolerance) const {
    if (ndim_ != other.ndim_) {
        throw std::invalid_argument(
            "Dimensionality of the two tensors must be the same");
    }

    if (dim > ndim_) {
        throw std::invalid_argument("Invalid concatenation dimension");
    }

    for (long d = 0; d < ndim_; ++d) {
        if (d != dim && size_[d] != other.size_[d]) {
            throw std::invalid_argument(
                "Tensor sizes must match apart from the concatenation "
                "dimension");
        }
    }

    const long size_1 = size_[dim];
    const long size_2 = other.size_[dim];

    const tensorfact::TtTensor<Real> tensor_1 =
        this->AddZeroPaddingBack(dim, size_2);
    const tensorfact::TtTensor<Real> tensor_2 =
        other.AddZeroPaddingFront(dim, size_1);

    tensorfact::TtTensor<Real> tensor = tensor_1 + tensor_2;
    tensor.Round(relative_tolerance);

    return tensor;
}

template <typename Real>
Real tensorfact::TtTensor<Real>::Dot(
    const tensorfact::TtTensor<Real> &other) const {
    if (ndim_ != other.ndim_) {
        throw std::invalid_argument(
            "Two tensors must have the same dimensionality");
    }

    for (long d = 0; d < ndim_; ++d) {
        if (size_[d] != other.size_[d]) {
            throw std::invalid_argument("Two tensors must have the same size");
        }
    }

    std::vector<Real> temp1;

    for (long d = ndim_ - 1; d >= 0; --d) {
        const long m = other.rank_[d];
        const long mm = other.rank_[d + 1];

        const long n = rank_[d];
        const long nn = rank_[d + 1];

        std::vector<Real> other_slice(m * mm);
        std::vector<Real> slice(n * nn);

        std::vector<std::vector<Real>> temp2(size_[d]);

        if (d == ndim_ - 1) {
            // Kronecker product of the last cores
            for (long k = 0; k < size_[d]; ++k) {
                for (long j = 0; j < mm; ++j) {
                    for (long i = 0; i < m; ++i) {
                        other_slice[i + m * j] =
                            other.param_[other.LinearIndex(i, k, j, d)];
                    }
                }

                for (long j = 0; j < nn; ++j) {
                    for (long i = 0; i < n; ++i) {
                        slice[i + j * n] = param_[LinearIndex(i, k, j, d)];
                    }
                }

                temp2[k].resize(m * n);
                blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                           blas::Op::Trans, m, n, mm, static_cast<Real>(1),
                           other_slice.data(), m, slice.data(), n,
                           static_cast<Real>(0), temp2[k].data(), m);
            }
        } else {
            // multiplication by Kronecker product of the cores
            for (long k = 0; k < size_[d]; ++k) {
                for (long j = 0; j < mm; ++j) {
                    for (long i = 0; i < m; ++i) {
                        other_slice[i + m * j] =
                            other.param_[other.LinearIndex(i, k, j, d)];
                    }
                }

                for (long j = 0; j < nn; ++j) {
                    for (long i = 0; i < n; ++i) {
                        slice[i + j * n] = param_[LinearIndex(i, k, j, d)];
                    }
                }

                std::vector<Real> temp3(m * nn);
                blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                           blas::Op::NoTrans, m, nn, mm, static_cast<Real>(1),
                           other_slice.data(), m, temp1.data(), mm,
                           static_cast<Real>(0), temp3.data(), m);

                temp2[k].resize(m * n);
                blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                           blas::Op::Trans, m, n, nn, static_cast<Real>(1),
                           temp3.data(), m, slice.data(), n,
                           static_cast<Real>(0), temp2[k].data(), m);
            }
        }

        temp1.resize(m * n);
        for (long j = 0; j < n; ++j) {
            for (long i = 0; i < m; ++i) {
                Real sum = static_cast<Real>(0);
                for (long k = 0; k < size_[d]; ++k) {
                    sum += temp2[k][i + j * m];
                }

                temp1[i + j * m] = sum;
            }
        }
    }

    return temp1[0];
}

template <typename Real>
Real tensorfact::TtTensor<Real>::FrobeniusNorm() const {
    return std::sqrt(this->Dot(*this));
}

template <typename Real>
std::vector<Real> tensorfact::TtTensor<Real>::Full() const {
    std::vector<Real> full;

    for (long d = 0; d < ndim_; ++d) {
        long length = offset_[d + 1] - offset_[d];

        if (d == 0) {
            full.resize(length);
            for (int n = 0; n < length; ++n) {
                full[n] = param_[offset_[d] + n];
            }
        } else {
            std::vector<Real> core(length);
            for (int n = 0; n < length; ++n) {
                core[n] = param_[offset_[d] + n];
            }

            const long m = full.size() / rank_[d];
            const long n = size_[d] * rank_[d + 1];
            const long k = rank_[d];

            std::vector<Real> full_new(m * n);
            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                       blas::Op::NoTrans, m, n, k, 1, full.data(), m,
                       core.data(), k, 0, full_new.data(), m);

            full = std::move(full_new);
        }
    }

    return full;
}

template <typename Real>
long tensorfact::TtTensor<Real>::LinearIndex(long i, long j, long k,
                                             long d) const {
    return i + rank_[d] * (j + size_[d] * k) + offset_[d];
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::AddZeroPaddingBack(
    long dim, long pad) const {
    tensorfact::TtTensor<Real> tt_tensor;

    tt_tensor.ndim_ = ndim_;

    tt_tensor.size_ = size_;
    tt_tensor.size_[dim] += pad;

    tt_tensor.rank_ = rank_;

    tt_tensor.offset_.resize(ndim_ + 1);
    tt_tensor.offset_[0] = 0;
    for (long d = 0; d < ndim_; ++d) {
        tt_tensor.offset_[d + 1] =
            tt_tensor.offset_[d] +
            tt_tensor.rank_[d] * tt_tensor.size_[d] * tt_tensor.rank_[d + 1];
    }

    tt_tensor.param_.resize(tt_tensor.offset_[ndim_]);
    for (long d = 0; d < ndim_; ++d) {
        for (long k = 0; k < tt_tensor.rank_[d + 1]; ++k) {
            for (long j = 0; j < tt_tensor.size_[d]; ++j) {
                for (long i = 0; i < tt_tensor.rank_[d]; ++i) {
                    if (d == dim) {
                        tt_tensor.param_[tt_tensor.LinearIndex(i, j, k, d)] =
                            (j < size_[d]) ? param_[LinearIndex(i, j, k, d)]
                                           : static_cast<Real>(0);
                    } else {
                        tt_tensor.param_[tt_tensor.LinearIndex(i, j, k, d)] =
                            param_[LinearIndex(i, j, k, d)];
                    }
                }
            }
        }
    }

    return tt_tensor;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::AddZeroPaddingFront(
    long dim, long pad) const {
    tensorfact::TtTensor<Real> tt_tensor;

    tt_tensor.ndim_ = ndim_;

    tt_tensor.size_ = size_;
    tt_tensor.size_[dim] += pad;

    tt_tensor.rank_ = rank_;

    tt_tensor.offset_.resize(ndim_ + 1);
    tt_tensor.offset_[0] = 0;
    for (long d = 0; d < ndim_; ++d) {
        tt_tensor.offset_[d + 1] =
            tt_tensor.offset_[d] +
            tt_tensor.rank_[d] * tt_tensor.size_[d] * tt_tensor.rank_[d + 1];
    }

    tt_tensor.param_.resize(tt_tensor.offset_[ndim_]);
    for (long d = 0; d < ndim_; ++d) {
        for (long k = 0; k < tt_tensor.rank_[d + 1]; ++k) {
            for (long j = 0; j < tt_tensor.size_[d]; ++j) {
                for (long i = 0; i < tt_tensor.rank_[d]; ++i) {
                    if (d == dim) {
                        tt_tensor.param_[tt_tensor.LinearIndex(i, j, k, d)] =
                            (j < pad) ? static_cast<Real>(0)
                                      : param_[LinearIndex(i, j - pad, k, d)];
                    } else {
                        tt_tensor.param_[tt_tensor.LinearIndex(i, j, k, d)] =
                            param_[LinearIndex(i, j, k, d)];
                    }
                }
            }
        }
    }

    return tt_tensor;
}

// explicit instantiations

template class tensorfact::TtTensor<float>;
template class tensorfact::TtTensor<double>;
