#include "tensorfact/tt_tensor.hpp"

#include <blas.hh>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "utils.hpp"

tensorfact::TtTensor::TtTensor(int ndim, const std::vector<int> &size,
                               const std::vector<int> &rank,
                               const std::vector<double> &param)
    : ndim_(ndim), size_(size), rank_(rank), offset_(ndim + 1), param_(param) {
    if (ndim_ < 1) {
        throw std::invalid_argument("Dimension must be positive");
    }

    if (size_.size() != ndim_) {
        throw std::invalid_argument(
            "Length of the size vector is incompatible with dimension");
    }
    for (int d = 0; d < ndim_; ++d) {
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
    for (int d = 1; d < ndim_; ++d) {
        if (rank_[d] < 1) {
            throw std::invalid_argument(
                "Entries of the rank vector must be positive");
        }
    }

    offset_[0] = 0;
    for (int d = 0; d < ndim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    if (param_.size() != offset_[ndim_]) {
        throw std::invalid_argument(
            "Length of the parameter vector is incompatible with dimension, "
            "size and rank");
    }
}

double tensorfact::TtTensor::Entry(const std::vector<int> &index) const {
    auto temp = std::make_shared<std::vector<double>>();

    for (int d = ndim_ - 1; d >= 0; --d) {
        auto slice =
            std::make_shared<std::vector<double>>(rank_[d] * rank_[d + 1]);

        for (int j = 0; j < rank_[d + 1]; ++j) {
            for (int i = 0; i < rank_[d]; ++i) {
                slice->at(i + j * rank_[d]) =
                    param_[LinearIndex(i, index[d], j, d)];
            }
        }

        if (d == ndim_ - 1) {
            std::swap(temp, slice);
        } else {
            auto temp_new = std::make_shared<std::vector<double>>(rank_[d]);
            blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, rank_[d],
                       rank_[d + 1], 1.0, slice->data(), rank_[d], temp->data(),
                       1, 0.0, temp_new->data(), 1);
            std::swap(temp, temp_new);
        }
    }

    return temp->at(0);
}
void tensorfact::TtTensor::WriteToFile(const std::string &file_name) const {
    std::ofstream file(file_name);

    file << "TT Tensor" << std::endl;

    file << ndim_ << std::endl;

    for (int d = 0; d < ndim_; ++d) {
        file << size_[d] << std::endl;
    }

    for (int d = 0; d <= ndim_; ++d) {
        file << rank_[d] << std::endl;
    }

    file << std::scientific;

    for (int d = 0; d < ndim_; ++d) {
        for (int k = 0; k < rank_[d + 1]; ++k) {
            for (int j = 0; j < size_[d]; ++j) {
                for (int i = 0; i < rank_[d]; ++i) {
                    file << std::setprecision(17)
                         << param_[LinearIndex(i, j, k, d)] << std::endl;
                }
            }
        }
    }

    file << std::defaultfloat;
}

void tensorfact::TtTensor::ReadFromFile(const std::string &file_name) {
    std::ifstream file(file_name);

    {
        std::string line;
        std::getline(file, line);
        if (line.compare("TT Tensor") != 0) {
            throw std::runtime_error("File is missing TT tensor header");
        }
    }

    {
        std::string line;
        std::getline(file, line);
        std::istringstream line_stream(line);
        line_stream >> ndim_;
    }

    size_.resize(ndim_);
    for (int d = 0; d < ndim_; ++d) {
        std::string line;
        std::getline(file, line);
        std::istringstream line_stream(line);
        line_stream >> size_[d];
    }

    rank_.resize(ndim_ + 1);
    for (int d = 0; d <= ndim_; ++d) {
        std::string line;
        std::getline(file, line);
        std::istringstream line_stream(line);
        line_stream >> rank_[d];
    }

    offset_.resize(ndim_ + 1);
    offset_[0] = 0;
    for (int d = 0; d < ndim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    param_.resize(offset_[ndim_]);
    for (int d = 0; d < ndim_; ++d) {
        for (int k = 0; k < rank_[d + 1]; ++k) {
            for (int j = 0; j < size_[d]; ++j) {
                for (int i = 0; i < rank_[d]; ++i) {
                    std::string line;
                    std::getline(file, line);
                    std::istringstream line_stream(line);
                    line_stream >> param_[LinearIndex(i, j, k, d)];
                }
            }
        }
    }
}

tensorfact::TtTensor tensorfact::TtTensor::operator+(
    const TtTensor &other) const {
    if (ndim_ != other.ndim_) {
        throw std::invalid_argument("TT tensors must have same dimensionality");
    }

    for (int d = 0; d < ndim_; ++d) {
        if (size_[d] != other.size_[d]) {
            throw std::invalid_argument("TT tensors must have same size");
        }
    }

    // new ranks
    std::vector<int> rank_new(ndim_ + 1);
    rank_new[0] = 1;
    for (int d = 1; d < ndim_; ++d) {
        rank_new[d] = rank_[d] + other.rank_[d];
    }
    rank_new[ndim_] = 1;

    // new offsets
    std::vector<int> offset_new(ndim_ + 1);
    offset_new[0] = 0;
    for (int d = 0; d < ndim_; ++d) {
        offset_new[d + 1] =
            offset_new[d] + rank_new[d] * size_[d] * rank_new[d + 1];
    }

    // new parameters
    std::vector<double> param_new(offset_new[ndim_]);

    // first core
    for (int k = 0; k < rank_[1]; ++k) {
        for (int j = 0; j < size_[0]; ++j) {
            param_new[j + k * size_[0] + offset_new[0]] =
                param_[LinearIndex(0, j, k, 0)];
        }
    }

    for (int k = 0; k < other.rank_[1]; ++k) {
        for (int j = 0; j < size_[0]; ++j) {
            param_new[j + (k + rank_[1]) * size_[0] + offset_new[0]] =
                other.param_[other.LinearIndex(0, j, k, 0)];
        }
    }

    // middle cores
    for (int d = 1; d < ndim_ - 1; ++d) {
        for (int k = 0; k < rank_[d + 1]; ++k) {
            for (int j = 0; j < size_[d]; ++j) {
                for (int i = 0; i < rank_[d]; ++i) {
                    param_new[i + j * rank_new[d] + k * rank_new[d] * size_[d] +
                              offset_new[d]] = param_[LinearIndex(i, j, k, d)];
                }
            }
        }

        for (int k = 0; k < other.rank_[d + 1]; ++k) {
            for (int j = 0; j < size_[d]; ++j) {
                for (int i = 0; i < other.rank_[d]; ++i) {
                    param_new[i + rank_[d] + j * rank_new[d] +
                              (k + rank_[d + 1]) * rank_new[d] * size_[d] +
                              offset_new[d]] =
                        other.param_[other.LinearIndex(i, j, k, d)];
                }
            }
        }
    }

    // last core
    for (int j = 0; j < size_[ndim_ - 1]; ++j) {
        for (int i = 0; i < rank_[ndim_ - 1]; ++i) {
            param_new[i + j * rank_new[ndim_ - 1] + offset_new[ndim_ - 1]] =
                param_[LinearIndex(i, j, 0, ndim_ - 1)];
        }

        for (int i = 0; i < other.rank_[ndim_ - 1]; ++i) {
            param_new[i + rank_[ndim_ - 1] + j * rank_new[ndim_ - 1] +
                      offset_new[ndim_ - 1]] =
                other.param_[other.LinearIndex(i, j, 0, ndim_ - 1)];
        }
    }

    return TtTensor(ndim_, size_, rank_new, param_new);
}

tensorfact::TtTensor tensorfact::TtTensor::operator*(double alpha) const {
    std::vector<double> param_new(param_);

    for (int n = 0; n < offset_[1]; ++n) {
        param_new[n] *= alpha;
    }

    return TtTensor(ndim_, size_, rank_, param_new);
}

tensorfact::TtTensor tensorfact::TtTensor::operator-(
    const tensorfact::TtTensor &other) const {
    return *this + other * (-1.0);
}

tensorfact::TtTensor tensorfact::TtTensor::operator/(double alpha) const {
    if (std::abs(alpha) < std::numeric_limits<double>::epsilon()) {
        throw std::logic_error("Dividing by a value too close to zero");
    }

    return *this * (1.0 / alpha);
}

void tensorfact::TtTensor::ComputeFromFull(const std::vector<int> &size,
                                           const std::vector<double> &array,
                                           double relative_tolerance) {
    if (relative_tolerance <= std::numeric_limits<double>::epsilon()) {
        throw std::logic_error("Required accuracy is too small");
    }

    ndim_ = size.size();
    size_ = size;
    rank_.resize(ndim_ + 1);

    const double delta = relative_tolerance / std::sqrt(ndim_ - 1);

    // compute cores
    rank_[0] = 1;

    std::vector<std::vector<double>> core(ndim_);
    core[ndim_ - 1] = array;

    for (int d = 0; d < ndim_ - 1; ++d) {
        const int m = rank_[d] * size_[d];
        const int n = core[ndim_ - 1].size() / m;

        int r;
        std::vector<double> s;
        std::vector<double> Vt;
        TruncatedSvd(m, n, core[ndim_ - 1], delta, true, r, core[d], s, Vt);

        core[ndim_ - 1].resize(r * n);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < r; ++i) {
                core[ndim_ - 1][i + j * r] = s[i] * Vt[i + j * r];
            }
        }

        rank_[d + 1] = r;
    }

    rank_[ndim_] = 1;

    // calculate offsets
    offset_.resize(ndim_ + 1);
    offset_[0] = 0;
    for (int d = 0; d < ndim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    // combine cores
    param_.resize(offset_[ndim_]);
    for (int d = 0; d < ndim_; ++d) {
        for (int k = 0; k < rank_[d + 1]; ++k) {
            for (int j = 0; j < size_[d]; ++j) {
                for (int i = 0; i < rank_[d]; ++i) {
                    param_[LinearIndex(i, j, k, d)] =
                        core[d][i + j * rank_[d] + k * rank_[d] * size_[d]];
                }
            }
        }
    }
}

void tensorfact::TtTensor::Round(double relative_tolerance) {
    // create cores
    std::vector<std::vector<double>> core(ndim_);
    for (int d = 0; d < ndim_; ++d) {
        core[d].resize(rank_[d] * size_[d] * rank_[d + 1]);
        for (int k = 0; k < rank_[d + 1]; ++k) {
            for (int j = 0; j < size_[d]; ++j) {
                for (int i = 0; i < rank_[d]; ++i) {
                    core[d][i + j * rank_[d] + k * rank_[d] * size_[d]] =
                        param_[LinearIndex(i, j, k, d)];
                }
            }
        }
    }

    // right-to-left orthogonalization
    for (int d = ndim_ - 1; d > 0; --d) {
        const int m = rank_[d];
        const int n = size_[d] * rank_[d + 1];
        const int k = std::min(m, n);

        std::vector<double> R;
        std::vector<double> Q;
        ThinRq(m, n, core[d], R, Q);

        core[d] = Q;

        std::vector<double> temp = core[d - 1];

        const int mm = rank_[d - 1] * size_[d - 1];
        core[d - 1].resize(mm * k);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   mm, k, m, 1.0, temp.data(), mm, R.data(), m, 0.0,
                   core[d - 1].data(), mm);

        rank_[d] = k;
    }

    // left-to-right compression
    if (relative_tolerance > 1.0e-15) {
        const double delta = relative_tolerance / std::sqrt(ndim_ - 1);

        for (int d = 0; d < ndim_ - 1; ++d) {
            const int m = rank_[d] * size_[d];
            const int n = rank_[d + 1];

            std::vector<double> U;
            std::vector<double> s;
            std::vector<double> Vt;
            int r;
            TruncatedSvd(m, n, core[d], delta, true, r, U, s, Vt);

            core[d] = U;

            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < r; ++i) {
                    Vt[i + j * r] *= s[i];
                }
            }

            std::vector<double> temp = core[d + 1];

            const int nn = size_[d + 1] * rank_[d + 2];
            core[d + 1].resize(r * nn);
            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                       blas::Op::NoTrans, r, nn, n, 1.0, Vt.data(), r,
                       temp.data(), n, 0.0, core[d + 1].data(), r);

            rank_[d + 1] = r;
        }
    }

    // recalculate offsets
    offset_.resize(ndim_ + 1);
    offset_[0] = 0;
    for (int d = 0; d < ndim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    // combine cores
    param_.resize(offset_[ndim_]);
    for (int d = 0; d < ndim_; ++d) {
        for (int k = 0; k < rank_[d + 1]; ++k) {
            for (int j = 0; j < size_[d]; ++j) {
                for (int i = 0; i < rank_[d]; ++i) {
                    param_[LinearIndex(i, j, k, d)] =
                        core[d][i + j * rank_[d] + k * rank_[d] * size_[d]];
                }
            }
        }
    }
}

tensorfact::TtTensor tensorfact::TtTensor::Concatenate(
    const tensorfact::TtTensor &other, int dim,
    double relative_tolerance) const {
    if (ndim_ != other.ndim_) {
        throw std::invalid_argument(
            "Dimensionality of the two tensors must be the same");
    }

    if (dim > ndim_) {
        throw std::invalid_argument("Invalid concatenation dimension");
    }

    for (int d = 0; d < ndim_; ++d) {
        if (d != dim && size_[d] != other.size_[d]) {
            throw std::invalid_argument(
                "Tensor sizes must match apart from the concatenation "
                "dimension");
        }
    }

    const int size_1 = size_[dim];
    const int size_2 = other.size_[dim];

    const tensorfact::TtTensor tensor_1 = this->AddZeroPaddingBack(dim, size_2);
    const tensorfact::TtTensor tensor_2 =
        other.AddZeroPaddingFront(dim, size_1);

    tensorfact::TtTensor tensor = tensor_1 + tensor_2;
    tensor.Round(relative_tolerance);

    return tensor;
}

double tensorfact::TtTensor::Dot(const tensorfact::TtTensor &other) const {
    if (ndim_ != other.ndim_) {
        throw std::invalid_argument(
            "Two tensors must have the same dimensionality");
    }

    for (int d = 0; d < ndim_; ++d) {
        if (size_[d] != other.size_[d]) {
            throw std::invalid_argument("Two tensors must have the same size");
        }
    }

    std::vector<double> temp1;

    for (int d = ndim_ - 1; d >= 0; --d) {
        const int m = other.rank_[d];
        const int mm = other.rank_[d + 1];

        const int n = rank_[d];
        const int nn = rank_[d + 1];

        std::vector<double> other_slice(m * mm);
        std::vector<double> slice(n * nn);

        std::vector<std::vector<double>> temp2(size_[d]);

        if (d == ndim_ - 1) {
            // Kronecker product of the last cores
            for (int k = 0; k < size_[d]; ++k) {
                for (int j = 0; j < mm; ++j) {
                    for (int i = 0; i < m; ++i) {
                        other_slice[i + m * j] =
                            other.param_[other.LinearIndex(i, k, j, d)];
                    }
                }

                for (int j = 0; j < nn; ++j) {
                    for (int i = 0; i < n; ++i) {
                        slice[i + j * n] = param_[LinearIndex(i, k, j, d)];
                    }
                }

                temp2[k].resize(m * n);
                blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                           blas::Op::Trans, m, n, mm, 1.0, other_slice.data(),
                           m, slice.data(), n, 0.0, temp2[k].data(), m);
            }
        } else {
            // multiplication by Kronecker product of the cores
            for (int k = 0; k < size_[d]; ++k) {
                for (int j = 0; j < mm; ++j) {
                    for (int i = 0; i < m; ++i) {
                        other_slice[i + m * j] =
                            other.param_[other.LinearIndex(i, k, j, d)];
                    }
                }

                for (int j = 0; j < nn; ++j) {
                    for (int i = 0; i < n; ++i) {
                        slice[i + j * n] = param_[LinearIndex(i, k, j, d)];
                    }
                }

                std::vector<double> temp3(m * nn);
                blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                           blas::Op::NoTrans, m, nn, mm, 1.0,
                           other_slice.data(), m, temp1.data(), mm, 0.0,
                           temp3.data(), m);

                temp2[k].resize(m * n);
                blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                           blas::Op::Trans, m, n, nn, 1.0, temp3.data(), m,
                           slice.data(), n, 0.0, temp2[k].data(), m);
            }
        }

        temp1.resize(m * n);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                double sum = 0.0;
                for (int k = 0; k < size_[d]; ++k) {
                    sum += temp2[k][i + j * m];
                }

                temp1[i + j * m] = sum;
            }
        }
    }

    return temp1[0];
}

double tensorfact::TtTensor::FrobeniusNorm() const {
    return std::sqrt(this->Dot(*this));
}

int tensorfact::TtTensor::LinearIndex(int i, int j, int k, int d) const {
    return i + rank_[d] * (j + size_[d] * k) + offset_[d];
}

tensorfact::TtTensor tensorfact::TtTensor::AddZeroPaddingBack(int dim,
                                                              int pad) const {
    tensorfact::TtTensor tt_tensor;

    tt_tensor.ndim_ = ndim_;

    tt_tensor.size_ = size_;
    tt_tensor.size_[dim] += pad;

    tt_tensor.rank_ = rank_;

    tt_tensor.offset_.resize(ndim_ + 1);
    tt_tensor.offset_[0] = 0;
    for (int d = 0; d < ndim_; ++d) {
        tt_tensor.offset_[d + 1] =
            tt_tensor.offset_[d] +
            tt_tensor.rank_[d] * tt_tensor.size_[d] * tt_tensor.rank_[d + 1];
    }

    tt_tensor.param_.resize(tt_tensor.offset_[ndim_]);
    for (int d = 0; d < ndim_; ++d) {
        for (int k = 0; k < tt_tensor.rank_[d + 1]; ++k) {
            for (int j = 0; j < tt_tensor.size_[d]; ++j) {
                for (int i = 0; i < tt_tensor.rank_[d]; ++i) {
                    if (d == dim) {
                        tt_tensor.param_[tt_tensor.LinearIndex(i, j, k, d)] =
                            (j < size_[d]) ? param_[LinearIndex(i, j, k, d)]
                                           : 0.0;
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

tensorfact::TtTensor tensorfact::TtTensor::AddZeroPaddingFront(int dim,
                                                               int pad) const {
    tensorfact::TtTensor tt_tensor;

    tt_tensor.ndim_ = ndim_;

    tt_tensor.size_ = size_;
    tt_tensor.size_[dim] += pad;

    tt_tensor.rank_ = rank_;

    tt_tensor.offset_.resize(ndim_ + 1);
    tt_tensor.offset_[0] = 0;
    for (int d = 0; d < ndim_; ++d) {
        tt_tensor.offset_[d + 1] =
            tt_tensor.offset_[d] +
            tt_tensor.rank_[d] * tt_tensor.size_[d] * tt_tensor.rank_[d + 1];
    }

    tt_tensor.param_.resize(tt_tensor.offset_[ndim_]);
    for (int d = 0; d < ndim_; ++d) {
        for (int k = 0; k < tt_tensor.rank_[d + 1]; ++k) {
            for (int j = 0; j < tt_tensor.size_[d]; ++j) {
                for (int i = 0; i < tt_tensor.rank_[d]; ++i) {
                    if (d == dim) {
                        tt_tensor.param_[tt_tensor.LinearIndex(i, j, k, d)] =
                            (j < pad) ? 0.0
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
