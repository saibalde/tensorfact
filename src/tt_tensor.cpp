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

template <typename Real>
tensorfact::TtTensor<Real>::TtTensor(long num_dim, long size, long rank)
    : num_dim_(num_dim),
      size_(num_dim),
      rank_(num_dim + 1),
      offset_(num_dim + 1),
      param_() {
    if (num_dim_ < 1) {
        throw std::invalid_argument("Dimension must be positive");
    }

    if (size < 1) {
        throw std::invalid_argument(
            "Entries of the size vector must be positive");
    }

    if (rank < 1) {
        throw std::invalid_argument(
            "Entries of the rank vector must be positive");
    }

    for (long d = 0; d < num_dim_; ++d) {
        size_[d] = size;
    }

    rank_[0] = 1;
    for (long d = 1; d < num_dim_; ++d) {
        rank_[d] = rank;
    }
    rank_[num_dim_] = 1;

    offset_[0] = 0;
    for (long d = 0; d < num_dim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    param_.resize(offset_[num_dim_]);
}

template <typename Real>
tensorfact::TtTensor<Real>::TtTensor(long num_dim,
                                     const std::vector<long> &size,
                                     const std::vector<long> &rank)
    : num_dim_(num_dim),
      size_(size),
      rank_(rank),
      offset_(num_dim + 1),
      param_() {
    if (num_dim_ < 1) {
        throw std::invalid_argument("Dimension must be positive");
    }

    if (size_.size() != num_dim_) {
        throw std::invalid_argument(
            "Length of the size vector is incompatible with dimension");
    }
    for (long d = 0; d < num_dim_; ++d) {
        if (size_[d] < 1) {
            throw std::invalid_argument(
                "Entries of the size vector must be positive");
        }
    }

    if (rank_.size() != num_dim_ + 1) {
        throw std::invalid_argument(
            "Length of the rank vector is incompatible with dimension");
    }
    if (rank_[0] != 1 || rank_[num_dim_] != 1) {
        throw std::invalid_argument("Boundary ranks must be one");
    }
    for (long d = 1; d < num_dim_; ++d) {
        if (rank_[d] < 1) {
            throw std::invalid_argument(
                "Entries of the rank vector must be positive");
        }
    }

    offset_[0] = 0;
    for (long d = 0; d < num_dim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    param_.resize(offset_[num_dim_]);
}

template <typename Real>
tensorfact::TtTensor<Real>::TtTensor(long num_dim,
                                     const std::vector<long> &size,
                                     const std::vector<long> &rank,
                                     const std::vector<Real> &param)
    : num_dim_(num_dim),
      size_(size),
      rank_(rank),
      offset_(num_dim + 1),
      param_(param) {
    if (num_dim_ < 1) {
        throw std::invalid_argument("Dimension must be positive");
    }

    if (size_.size() != num_dim_) {
        throw std::invalid_argument(
            "Length of the size vector is incompatible with dimension");
    }
    for (long d = 0; d < num_dim_; ++d) {
        if (size_[d] < 1) {
            throw std::invalid_argument(
                "Entries of the size vector must be positive");
        }
    }

    if (rank_.size() != num_dim_ + 1) {
        throw std::invalid_argument(
            "Length of the rank vector is incompatible with dimension");
    }
    if (rank_[0] != 1 || rank_[num_dim_] != 1) {
        throw std::invalid_argument("Boundary ranks must be one");
    }
    for (long d = 1; d < num_dim_; ++d) {
        if (rank_[d] < 1) {
            throw std::invalid_argument(
                "Entries of the rank vector must be positive");
        }
    }

    offset_[0] = 0;
    for (long d = 0; d < num_dim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    if (param_.size() != offset_[num_dim_]) {
        throw std::invalid_argument(
            "Length of the parameter vector is incompatible with dimension, "
            "size and rank");
    }
}

template <typename Real>
long tensorfact::TtTensor<Real>::NumElement() const {
    long num_element = 1;

    for (long d = 0; d < num_dim_; ++d) {
        num_element *= size_[d];
    }

    return num_element;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator+=(
    const TtTensor<Real> &other) {
    if (num_dim_ != other.num_dim_) {
        throw std::invalid_argument("TT tensors must have same dimensionality");
    }

    for (long d = 0; d < num_dim_; ++d) {
        if (size_[d] != other.size_[d]) {
            throw std::invalid_argument("TT tensors must have same size");
        }
    }

    // copy current object
    tensorfact::TtTensor<Real> self(*this);

    // update ranks
    for (long d = 1; d < num_dim_; ++d) {
        rank_[d] = self.rank_[d] + other.rank_[d];
    }

    // update offsets
    for (long d = 0; d < num_dim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    // create new parameters
    param_.resize(offset_[num_dim_]);
    for (auto &p : param_) {
        p = 0;
    }

    // first core
    for (long k = 0; k < self.rank_[1]; ++k) {
        for (long j = 0; j < size_[0]; ++j) {
            Param(0, j, k, 0) = self.Param(0, j, k, 0);
        }
    }

    for (long k = 0; k < other.rank_[1]; ++k) {
        for (long j = 0; j < size_[0]; ++j) {
            Param(0, j, k + self.rank_[1], 0) = other.Param(0, j, k, 0);
        }
    }

    // middle cores
    for (long d = 1; d < num_dim_ - 1; ++d) {
        for (long k = 0; k < self.rank_[d + 1]; ++k) {
            for (long j = 0; j < size_[d]; ++j) {
                for (long i = 0; i < self.rank_[d]; ++i) {
                    Param(i, j, k, d) = self.Param(i, j, k, d);
                }
            }
        }

        for (long k = 0; k < other.rank_[d + 1]; ++k) {
            for (long j = 0; j < size_[d]; ++j) {
                for (long i = 0; i < other.rank_[d]; ++i) {
                    Param(i + self.rank_[d], j, k + self.rank_[d + 1], d) =
                        other.Param(i, j, k, d);
                }
            }
        }
    }

    // last core
    for (long j = 0; j < size_[num_dim_ - 1]; ++j) {
        for (long i = 0; i < self.rank_[num_dim_ - 1]; ++i) {
            Param(i, j, 0, num_dim_ - 1) = self.Param(i, j, 0, num_dim_ - 1);
        }

        for (long i = 0; i < other.rank_[num_dim_ - 1]; ++i) {
            Param(i + self.rank_[num_dim_ - 1], j, 0, num_dim_ - 1) =
                other.Param(i, j, 0, num_dim_ - 1);
        }
    }

    return *this;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator-=(
    const tensorfact::TtTensor<Real> &other) {
    *this += other * static_cast<Real>(-1);
    return *this;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator*=(Real alpha) {
    for (long k = 0; k < rank_[1]; ++k) {
        for (long j = 0; j < size_[0]; ++j) {
            Param(0, j, k, 0) *= alpha;
        }
    }

    return *this;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator/=(Real alpha) {
    if (std::abs(alpha) < std::numeric_limits<Real>::epsilon()) {
        throw std::logic_error("Dividing by a value too close to zero");
    }

    *this *= 1 / alpha;
    return *this;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator*=(
    const TtTensor<Real> &other) {
    if (num_dim_ != other.num_dim_) {
        throw std::invalid_argument("TT tensors must have same dimensionality");
    }

    for (long d = 0; d < num_dim_; ++d) {
        if (size_[d] != other.size_[d]) {
            throw std::invalid_argument("TT tensors must have same size");
        }
    }

    // copy current object
    tensorfact::TtTensor<Real> self(*this);

    // update ranks
    for (long d = 0; d <= num_dim_; ++d) {
        rank_[d] = self.rank_[d] * other.rank_[d];
    }

    // update offsets
    for (long d = 0; d < num_dim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    // create new parameters
    param_.resize(offset_[num_dim_]);

    // compute cores
    for (long d = 0; d < num_dim_; ++d) {
        for (long k = 0; k < rank_[d + 1]; ++k) {
            for (long j = 0; j < size_[d]; ++j) {
                for (long i = 0; i < rank_[d]; ++i) {
                    const long i1 = i / other.rank_[d];
                    const long i2 = i % other.rank_[d];

                    const long k1 = k / other.rank_[d + 1];
                    const long k2 = k % other.rank_[d + 1];

                    Param(i, j, k, d) =
                        self.Param(i1, j, k1, d) * other.Param(i2, j, k2, d);
                }
            }
        }
    }

    return *this;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator+(
    const tensorfact::TtTensor<Real> &other) const {
    tensorfact::TtTensor<Real> self(*this);
    self += other;
    return self;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator-(
    const tensorfact::TtTensor<Real> &other) const {
    tensorfact::TtTensor<Real> self(*this);
    self -= other;
    return self;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator*(
    Real alpha) const {
    tensorfact::TtTensor<Real> self(*this);
    self *= alpha;
    return self;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator/(
    Real alpha) const {
    tensorfact::TtTensor<Real> self(*this);
    self /= alpha;
    return self;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator*(
    const TtTensor<Real> &other) const {
    tensorfact::TtTensor<Real> self(*this);
    self *= other;
    return self;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::Concatenate(
    const tensorfact::TtTensor<Real> &other, long dim,
    Real relative_tolerance) const {
    if (num_dim_ != other.num_dim_) {
        throw std::invalid_argument(
            "Dimensionality of the two tensors must be the same");
    }

    if (dim > num_dim_) {
        throw std::invalid_argument("Invalid concatenation dimension");
    }

    for (long d = 0; d < num_dim_; ++d) {
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
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::Shift(long d,
                                                             long shift) const {
    if (d < 0 || d >= num_dim_) {
        throw std::invalid_argument("Specified dimension is invalid");
    }

    tensorfact::TtTensor<Real> self(*this);

    for (long k = 0; k < rank_[d + 1]; ++k) {
        for (long j = 0; j < size_[d]; ++j) {
            for (long i = 0; i < rank_[d]; ++i) {
                if (shift >= 0) {
                    self.Param(i, j, k, d) = (j < size_[d] - shift)
                                                 ? Param(i, j + shift, k, d)
                                                 : 0.0;
                } else {
                    self.Param(i, j, k, d) =
                        (j >= -shift) ? Param(i, j + shift, k, d) : 0.0;
                }
            }
        }
    }

    return self;
}

template <typename Real>
Real tensorfact::TtTensor<Real>::Dot(
    const tensorfact::TtTensor<Real> &other) const {
    if (num_dim_ != other.num_dim_) {
        throw std::invalid_argument(
            "Two tensors must have the same dimensionality");
    }

    for (long d = 0; d < num_dim_; ++d) {
        if (size_[d] != other.size_[d]) {
            throw std::invalid_argument("Two tensors must have the same size");
        }
    }

    std::vector<Real> temp1;

    for (long d = num_dim_ - 1; d >= 0; --d) {
        const long m = other.rank_[d];
        const long mm = other.rank_[d + 1];

        const long n = rank_[d];
        const long nn = rank_[d + 1];

        std::vector<Real> other_slice(m * mm);
        std::vector<Real> slice(n * nn);

        std::vector<std::vector<Real>> temp2(size_[d]);

        if (d == num_dim_ - 1) {
            // Kronecker product of the last cores
            for (long k = 0; k < size_[d]; ++k) {
                for (long j = 0; j < mm; ++j) {
                    for (long i = 0; i < m; ++i) {
                        other_slice[i + m * j] = other.Param(i, k, j, d);
                    }
                }

                for (long j = 0; j < nn; ++j) {
                    for (long i = 0; i < n; ++i) {
                        slice[i + j * n] = Param(i, k, j, d);
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
                        other_slice[i + m * j] = other.Param(i, k, j, d);
                    }
                }

                for (long j = 0; j < nn; ++j) {
                    for (long i = 0; i < n; ++i) {
                        slice[i + j * n] = Param(i, k, j, d);
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
void tensorfact::TtTensor<Real>::Round(Real relative_tolerance) {
    // create cores
    std::vector<std::vector<Real>> core(num_dim_);
    for (long d = 0; d < num_dim_; ++d) {
        core[d].resize(rank_[d] * size_[d] * rank_[d + 1]);
        for (long k = 0; k < rank_[d + 1]; ++k) {
            for (long j = 0; j < size_[d]; ++j) {
                for (long i = 0; i < rank_[d]; ++i) {
                    core[d][i + j * rank_[d] + k * rank_[d] * size_[d]] =
                        Param(i, j, k, d);
                }
            }
        }
    }

    // right-to-left orthogonalization
    for (long d = num_dim_ - 1; d > 0; --d) {
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
            relative_tolerance / std::sqrt(static_cast<Real>(num_dim_ - 1));

        for (long d = 0; d < num_dim_ - 1; ++d) {
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
    offset_.resize(num_dim_ + 1);
    offset_[0] = 0;
    for (long d = 0; d < num_dim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    // combine cores
    param_.resize(offset_[num_dim_]);
    for (long d = 0; d < num_dim_; ++d) {
        for (long k = 0; k < rank_[d + 1]; ++k) {
            for (long j = 0; j < size_[d]; ++j) {
                for (long i = 0; i < rank_[d]; ++i) {
                    Param(i, j, k, d) =
                        core[d][i + j * rank_[d] + k * rank_[d] * size_[d]];
                }
            }
        }
    }
}

template <typename Real>
std::vector<Real> tensorfact::TtTensor<Real>::Full() const {
    std::vector<Real> full;

    for (long d = 0; d < num_dim_; ++d) {
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
Real tensorfact::TtTensor<Real>::Entry(const std::vector<long> &index) const {
    auto temp = std::make_shared<std::vector<Real>>();

    for (long d = num_dim_ - 1; d >= 0; --d) {
        auto slice =
            std::make_shared<std::vector<Real>>(rank_[d] * rank_[d + 1]);

        for (long j = 0; j < rank_[d + 1]; ++j) {
            for (long i = 0; i < rank_[d]; ++i) {
                slice->at(i + j * rank_[d]) = Param(i, index[d], j, d);
            }
        }

        if (d == num_dim_ - 1) {
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
    std::ofstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing TT tensor");
    }

    file << "TT Tensor" << std::endl;
    file << std::endl;

    file << "num_dim" << std::endl;
    file << num_dim_ << std::endl;
    file << std::endl;

    file << "size" << std::endl;
    for (long d = 0; d < num_dim_; ++d) {
        file << size_[d] << std::endl;
    }
    file << std::endl;

    file << "rank" << std::endl;
    for (long d = 0; d <= num_dim_; ++d) {
        file << rank_[d] << std::endl;
    }
    file << std::endl;

    file << "parameter" << std::endl;
    file << std::scientific;
    for (long n = 0; n < offset_[num_dim_]; ++n) {
        file << std::setw(24) << std::setprecision(17) << param_[n]
             << std::endl;
    }
    file << std::defaultfloat;
    file << std::endl;
}

template <typename Real>
void tensorfact::TtTensor<Real>::ReadFromFile(const std::string &file_name) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading TT tensor");
    }

    std::string line;

    {
        std::getline(file, line);
        if (line.compare("TT Tensor") != 0) {
            throw std::runtime_error("File is missing TT tensor header");
        }
    }

    std::getline(file, line);

    {
        std::getline(file, line);
        if (line.compare("num_dim") != 0) {
            throw std::runtime_error(
                "File does not specify number of dimensions");
        }
    }

    {
        std::getline(file, line);
        std::istringstream line_stream(line);
        line_stream >> num_dim_;

        if (num_dim_ < 1) {
            throw std::runtime_error(
                "File specifies invalid number of dimensions");
        }
    }

    std::getline(file, line);

    size_.resize(num_dim_);
    rank_.resize(num_dim_ + 1);
    offset_.resize(num_dim_ + 1);

    {
        std::getline(file, line);
        if (line.compare("size") != 0) {
            throw std::runtime_error("File does not specify tensor mode sizes");
        }
    }

    for (long d = 0; d < num_dim_; ++d) {
        if (std::getline(file, line)) {
            std::istringstream line_stream(line);
            line_stream >> size_[d];

            if (size_[d] < 1) {
                throw std::runtime_error(
                    "File specifies invalid tensor mode size");
            }
        } else {
            throw std::runtime_error("Reached end of file unexpectedly");
        }
    }

    std::getline(file, line);

    {
        std::getline(file, line);
        if (line.compare("rank") != 0) {
            throw std::runtime_error("File does not specify TT ranks");
        }
    }

    for (long d = 0; d <= num_dim_; ++d) {
        if (std::getline(file, line)) {
            std::istringstream line_stream(line);
            line_stream >> rank_[d];

            if ((d == 0 || d == num_dim_) && rank_[d] != 1) {
                throw std::runtime_error(
                    "File specifies invalid TT boundary rank");
            } else if ((0 < d && d < num_dim_) && rank_[d] < 1) {
                throw std::runtime_error(
                    "File specifies invalid TT interior rank");
            }
        } else {
            throw std::runtime_error("Reached end of file unexpectedly");
        }
    }

    std::getline(file, line);

    offset_[0] = 0;
    for (long d = 0; d < num_dim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    param_.resize(offset_[num_dim_]);

    {
        std::getline(file, line);
        if (line.compare("parameter") != 0) {
            throw std::runtime_error("File does not specify TT parameters");
        }
    }
    
    for (long n = 0; n < offset_[num_dim_]; ++n) {
        if (std::getline(file, line)) {
            std::istringstream line_stream(line);
            line_stream >> param_[n];
        } else {
            throw std::runtime_error("Reached end of file unexpectedly");
        }
    }

    std::getline(file, line);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::AddZeroPaddingBack(
    long dim, long pad) const {
    tensorfact::TtTensor<Real> tt_tensor;

    tt_tensor.num_dim_ = num_dim_;

    tt_tensor.size_ = size_;
    tt_tensor.size_[dim] += pad;

    tt_tensor.rank_ = rank_;

    tt_tensor.offset_.resize(num_dim_ + 1);
    tt_tensor.offset_[0] = 0;
    for (long d = 0; d < num_dim_; ++d) {
        tt_tensor.offset_[d + 1] =
            tt_tensor.offset_[d] +
            tt_tensor.rank_[d] * tt_tensor.size_[d] * tt_tensor.rank_[d + 1];
    }

    tt_tensor.param_.resize(tt_tensor.offset_[num_dim_]);
    for (long d = 0; d < num_dim_; ++d) {
        for (long k = 0; k < tt_tensor.rank_[d + 1]; ++k) {
            for (long j = 0; j < tt_tensor.size_[d]; ++j) {
                for (long i = 0; i < tt_tensor.rank_[d]; ++i) {
                    if (d == dim) {
                        tt_tensor.Param(i, j, k, d) =
                            (j < size_[d]) ? Param(i, j, k, d) : 0;
                    } else {
                        tt_tensor.Param(i, j, k, d) = Param(i, j, k, d);
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

    tt_tensor.num_dim_ = num_dim_;

    tt_tensor.size_ = size_;
    tt_tensor.size_[dim] += pad;

    tt_tensor.rank_ = rank_;

    tt_tensor.offset_.resize(num_dim_ + 1);
    tt_tensor.offset_[0] = 0;
    for (long d = 0; d < num_dim_; ++d) {
        tt_tensor.offset_[d + 1] =
            tt_tensor.offset_[d] +
            tt_tensor.rank_[d] * tt_tensor.size_[d] * tt_tensor.rank_[d + 1];
    }

    tt_tensor.param_.resize(tt_tensor.offset_[num_dim_]);
    for (long d = 0; d < num_dim_; ++d) {
        for (long k = 0; k < tt_tensor.rank_[d + 1]; ++k) {
            for (long j = 0; j < tt_tensor.size_[d]; ++j) {
                for (long i = 0; i < tt_tensor.rank_[d]; ++i) {
                    if (d == dim) {
                        tt_tensor.Param(i, j, k, d) =
                            (j < pad) ? 0 : Param(i, j - pad, k, d);
                    } else {
                        tt_tensor.Param(i, j, k, d) = Param(i, j, k, d);
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
