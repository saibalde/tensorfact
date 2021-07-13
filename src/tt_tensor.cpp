#include "tensorfact/tt_tensor.hpp"

#include <blas.hh>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "thin_rq.hpp"
#include "truncated_svd.hpp"

// implementation

template <typename Real>
class tensorfact::TtTensor<Real>::Impl {
public:
    Impl() = default;

    Impl(long num_dim, const std::vector<long> &size,
         const std::vector<long> &rank, const std::vector<Real> &param);

    Impl(const Impl &) = default;

    Impl(Impl &&) = default;

    ~Impl() = default;

    Impl &operator=(const Impl &) = default;

    Impl &operator=(Impl &&) = default;

    const long &NumDim() const { return num_dim_; }

    const std::vector<long> &Size() const { return size_; }

    const std::vector<long> &Rank() const { return rank_; }

    const std::vector<Real> &Param() const { return param_; }

    std::vector<Real> &Param() { return param_; }

    const Real &Param(long i, long j, long k, long d) const {
        return param_[LinearIndex(i, j, k, d)];
    }

    Real &Param(long i, long j, long k, long d) {
        return param_[LinearIndex(i, j, k, d)];
    }

    const long &NumParam() const { return offset_[num_dim_]; }

    long NumElement() const;

    Impl operator+=(const Impl &other);

    Impl operator-=(const Impl &other);

    Impl operator*=(Real alpha);

    Impl operator/=(Real alpha);

    Impl operator*=(const Impl &other);

    Impl operator+(const Impl &other) const;

    Impl operator-(const Impl &other) const;

    Impl operator*(Real alpha) const;

    Impl operator/(Real alpha) const;

    Impl operator*(const Impl &other) const;

    Impl Concatenate(const Impl &other, long d) const;

    Impl Shift(long d, long shift) const;

    Real Contract(const std::vector<std::vector<Real>> &vectors) const;

    Real Dot(const Impl &other) const;

    Real FrobeniusNorm() const;

    void Round(Real relative_tolerance);

    std::vector<Real> Full() const;

    Real Entry(const std::vector<long> &index) const;

    void WriteToFile(const std::string &file_name) const;

    void ReadFromFile(const std::string &file_name);

private:
    long LinearIndex(long i, long j, long k, long d) const {
        return i + rank_[d] * (j + size_[d] * k) + offset_[d];
    }

    Impl AddZeroPaddingBack(long dim, long pad) const;

    Impl AddZeroPaddingFront(long dim, long pad) const;

    long num_dim_;
    std::vector<long> size_;
    std::vector<long> rank_;
    std::vector<long> offset_;
    std::vector<Real> param_;
};

template <typename Real>
tensorfact::TtTensor<Real>::Impl::Impl(long num_dim,
                                       const std::vector<long> &size,
                                       const std::vector<long> &rank,
                                       const std::vector<Real> &param)
    : num_dim_(num_dim),
      size_(size),
      rank_(rank),
      offset_(num_dim + 1),
      param_(param) {
    if (num_dim_ < 1) {
        throw std::invalid_argument("Number of dimensions must be positive");
    }

    if (size_.size() != num_dim_) {
        throw std::invalid_argument(
            "Size vector length must equal number of dimensions");
    }

    for (long d = 0; d < num_dim_; ++d) {
        if (size_[d] < 1) {
            throw std::invalid_argument("Size vector entries must be positive");
        }
    }

    if (rank_.size() != num_dim_ + 1) {
        throw std::invalid_argument(
            "Rank vector length must equal one plus number of dimensions");
    }

    if (rank_[0] != 1 || rank_[num_dim_] != 1) {
        throw std::invalid_argument("Boundary ranks must be one");
    }

    for (long d = 1; d < num_dim_; ++d) {
        if (rank_[d] < 1) {
            throw std::invalid_argument("Interior ranks must be positive");
        }
    }

    offset_[0] = 0;
    for (long d = 0; d < num_dim_; ++d) {
        offset_[d + 1] = offset_[d] + rank_[d] * size_[d] * rank_[d + 1];
    }

    if (param_.size() != offset_[num_dim_]) {
        throw std::invalid_argument(
            "Parameter vector length is incompatible with number of "
            "dimensions, size and rank");
    }
}

template <typename Real>
long tensorfact::TtTensor<Real>::Impl::NumElement() const {
    long num_element = 1;

    for (long d = 0; d < num_dim_; ++d) {
        num_element *= size_[d];
    }

    return num_element;
}

template <typename Real>
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::operator+=(
    const tensorfact::TtTensor<Real>::Impl &other) {
    if (num_dim_ != other.num_dim_) {
        throw std::invalid_argument(
            "TT tensors must have same number of dimensions");
    }

    for (long d = 0; d < num_dim_; ++d) {
        if (size_[d] != other.size_[d]) {
            throw std::invalid_argument("TT tensors must have same size");
        }
    }

    // copy current object
    tensorfact::TtTensor<Real>::Impl self(*this);

    // update ranks
    for (long d = 1; d < num_dim_; ++d) {
        rank_[d] = self.rank_[d] + other.rank_[d];
    }

    // update offsets
    offset_[0] = 0;
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
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::operator-=(
    const tensorfact::TtTensor<Real>::Impl &other) {
    *this += other * static_cast<Real>(-1);
    return *this;
}

template <typename Real>
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::operator*=(Real alpha) {
    for (long k = 0; k < rank_[1]; ++k) {
        for (long j = 0; j < size_[0]; ++j) {
            Param(0, j, k, 0) *= alpha;
        }
    }

    return *this;
}

template <typename Real>
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::operator/=(Real alpha) {
    *this *= 1 / alpha;
    return *this;
}

template <typename Real>
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::operator*=(
    const TtTensor<Real>::Impl &other) {
    if (num_dim_ != other.num_dim_) {
        throw std::invalid_argument(
            "TT tensors must have same number of dimensions");
    }

    for (long d = 0; d < num_dim_; ++d) {
        if (size_[d] != other.size_[d]) {
            throw std::invalid_argument("TT tensors must have same size");
        }
    }

    // copy current object
    tensorfact::TtTensor<Real>::Impl self(*this);

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
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::operator+(
    const tensorfact::TtTensor<Real>::Impl &other) const {
    tensorfact::TtTensor<Real>::Impl self(*this);
    self += other;
    return self;
}

template <typename Real>
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::operator-(
    const tensorfact::TtTensor<Real>::Impl &other) const {
    tensorfact::TtTensor<Real>::Impl self(*this);
    self -= other;
    return self;
}

template <typename Real>
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::operator*(Real alpha) const {
    tensorfact::TtTensor<Real>::Impl self(*this);
    self *= alpha;
    return self;
}

template <typename Real>
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::operator/(Real alpha) const {
    tensorfact::TtTensor<Real>::Impl self(*this);
    self /= alpha;
    return self;
}

template <typename Real>
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::operator*(
    const TtTensor<Real>::Impl &other) const {
    tensorfact::TtTensor<Real>::Impl self(*this);
    self *= other;
    return self;
}

template <typename Real>
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::Concatenate(
    const tensorfact::TtTensor<Real>::Impl &other, long dim) const {
    if (num_dim_ != other.num_dim_) {
        throw std::invalid_argument(
            "Two tensors must have same number of dimensions");
    }

    if (dim < 0 || dim > num_dim_) {
        throw std::invalid_argument("Invalid concatenation dimension");
    }

    for (long d = 0; d < num_dim_; ++d) {
        if (d != dim && size_[d] != other.size_[d]) {
            throw std::invalid_argument(
                "Tensor sizes must match apart from the concatenation "
                "dimension");
        }
    }

    if (dim < num_dim_) {
        const long size_1 = size_[dim];
        const long size_2 = other.size_[dim];

        const tensorfact::TtTensor<Real>::Impl tensor_1 =
            AddZeroPaddingBack(dim, size_2);
        const tensorfact::TtTensor<Real>::Impl tensor_2 =
            other.AddZeroPaddingFront(dim, size_1);

        tensorfact::TtTensor<Real>::Impl tensor = tensor_1 + tensor_2;

        return tensor;
    }

    long num_dim = num_dim_ + 1;

    std::vector<long> size(num_dim);
    for (long d = 0; d < num_dim_; ++d) {
        size[d] = size_[d];
    }
    size[num_dim_] = 2;

    std::vector<long> rank(num_dim + 1);
    rank[0] = 1;
    for (long d = 1; d < num_dim_; ++d) {
        rank[d] = rank_[d] + other.rank_[d];
    }
    rank[num_dim_] = 2;
    rank[num_dim] = 1;

    long num_param = 0;
    for (long d = 0; d < num_dim; ++d) {
        num_param += rank[d] * size[d] * rank[d + 1];
    }

    std::vector<Real> param(num_param, 0);

    tensorfact::TtTensor<Real>::Impl tensor(num_dim, size, rank, param);

    for (long d = 0; d < num_dim; ++d) {
        if (d == 0) {
            for (long k = 0; k < rank_[d + 1]; ++k) {
                for (long j = 0; j < size[d]; ++j) {
                    tensor.Param(0, j, k, d) = Param(0, j, k, d);
                }
            }

            for (long k = 0; k < other.rank_[d + 1]; ++k) {
                for (long j = 0; j < size[d]; ++j) {
                    tensor.Param(0, j, k + rank_[d + 1], d) =
                        other.Param(0, j, k, d);
                }
            }
        } else if (d == num_dim_) {
            tensor.Param(0, 0, 0, d) = 1;
            tensor.Param(1, 0, 0, d) = 0;
            tensor.Param(0, 1, 0, d) = 0;
            tensor.Param(1, 1, 0, d) = 1;
        } else {
            for (long k = 0; k < rank_[d + 1]; ++k) {
                for (long j = 0; j < size[d]; ++j) {
                    for (long i = 0; i < rank_[d]; ++i) {
                        tensor.Param(i, j, k, d) = Param(i, j, k, d);
                    }
                }
            }

            for (long k = 0; k < other.rank_[d + 1]; ++k) {
                for (long j = 0; j < size[d]; ++j) {
                    for (long i = 0; i < other.rank_[d]; ++i) {
                        tensor.Param(i + rank_[d], j, k + rank_[d + 1], d) =
                            other.Param(i, j, k, d);
                    }
                }
            }
        }
    }

    return tensor;
}

template <typename Real>
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::Shift(long d, long shift) const {
    if (d < 0 || d >= num_dim_) {
        throw std::invalid_argument("Specified dimension is invalid");
    }

    tensorfact::TtTensor<Real>::Impl self(*this);

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
Real tensorfact::TtTensor<Real>::Impl::Contract(
    const std::vector<std::vector<Real>> &vectors) const {
    if (vectors.size() != num_dim_) {
        throw std::invalid_argument(
            "Number of vectors must match the dimensionality of the tensor");
    }

    for (long d = 0; d < num_dim_; ++d) {
        if (vectors[d].size() != size_[d]) {
            throw std::invalid_argument(
                "Size of the vector must match the corresponding tensor mode "
                "size");
        }
    }

    std::shared_ptr<std::vector<Real>> temp1 = nullptr;

    for (long d = num_dim_ - 1; d >= 0; --d) {
        if (d == num_dim_ - 1) {
            temp1 = std::make_shared<std::vector<Real>>(rank_[d], 0);

            for (long j = 0; j < size_[d]; ++j) {
                for (long i = 0; i < rank_[d]; ++i) {
                    temp1->at(i) += Param(i, j, 0, d) * vectors[d][j];
                }
            }
        } else {
            std::vector<Real> slice(rank_[d] * rank_[d + 1], 0);
            for (long k = 0; k < rank_[d + 1]; ++k) {
                for (long j = 0; j < size_[d]; ++j) {
                    for (long i = 0; i < rank_[d]; ++i) {
                        slice[i + rank_[d] * k] +=
                            Param(i, j, k, d) * vectors[d][j];
                    }
                }
            }

            auto temp2 = std::make_shared<std::vector<Real>>(rank_[d]);
            blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, rank_[d],
                       rank_[d + 1], 1, slice.data(), rank_[d], temp1->data(),
                       1, 0, temp2->data(), 1);
            std::swap(temp1, temp2);
        }
    }

    return temp1->at(0);
}

template <typename Real>
Real tensorfact::TtTensor<Real>::Impl::Dot(
    const tensorfact::TtTensor<Real>::Impl &other) const {
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
Real tensorfact::TtTensor<Real>::Impl::FrobeniusNorm() const {
    return std::sqrt(this->Dot(*this));
}

template <typename Real>
void tensorfact::TtTensor<Real>::Impl::Round(Real relative_tolerance) {
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
std::vector<Real> tensorfact::TtTensor<Real>::Impl::Full() const {
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
Real tensorfact::TtTensor<Real>::Impl::Entry(
    const std::vector<long> &index) const {
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
void tensorfact::TtTensor<Real>::Impl::WriteToFile(
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
void tensorfact::TtTensor<Real>::Impl::ReadFromFile(
    const std::string &file_name) {
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
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::AddZeroPaddingBack(long dim, long pad) const {
    tensorfact::TtTensor<Real>::Impl tt_tensor;

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
typename tensorfact::TtTensor<Real>::Impl
tensorfact::TtTensor<Real>::Impl::AddZeroPaddingFront(long dim,
                                                      long pad) const {
    tensorfact::TtTensor<Real>::Impl tt_tensor;

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

// forwarding

template <typename Real>
tensorfact::TtTensor<Real>::TtTensor() {
    impl_ = std::make_shared<tensorfact::TtTensor<Real>::Impl>();
}

template <typename Real>
tensorfact::TtTensor<Real>::TtTensor(long num_dim, long size, long rank) {
    std::vector<long> size_vector(num_dim, size);

    std::vector<long> rank_vector(num_dim + 1, rank);
    rank_vector[0] = 1;
    rank_vector[num_dim] = 1;

    long num_param = size * (2 * rank + (num_dim - 2) * rank * rank);
    std::vector<Real> param(num_param);

    impl_ = std::make_shared<tensorfact::TtTensor<Real>::Impl>(
        num_dim, size_vector, rank_vector, param);
}

template <typename Real>
tensorfact::TtTensor<Real>::TtTensor(long num_dim,
                                     const std::vector<long> &size,
                                     const std::vector<long> &rank) {
    long num_param = 0;
    for (long d = 0; d < num_dim; ++d) {
        num_param += rank[d] * size[d] * rank[d + 1];
    }
    std::vector<Real> param(num_param);

    impl_ = std::make_shared<tensorfact::TtTensor<Real>::Impl>(num_dim, size,
                                                               rank, param);
}

template <typename Real>
tensorfact::TtTensor<Real>::TtTensor(long num_dim,
                                     const std::vector<long> &size,
                                     const std::vector<long> &rank,
                                     const std::vector<Real> &param) {
    impl_ = std::make_shared<tensorfact::TtTensor<Real>::Impl>(num_dim, size,
                                                               rank, param);
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::Copy() const {
    tensorfact::TtTensor<Real> tt_tensor;
    *(tt_tensor.impl_) = *impl_;
    return tt_tensor;
}

template <typename Real>
const long &tensorfact::TtTensor<Real>::NumDim() const {
    return impl_->NumDim();
}

template <typename Real>
const std::vector<long> &tensorfact::TtTensor<Real>::Size() const {
    return impl_->Size();
}

template <typename Real>
const long &tensorfact::TtTensor<Real>::Size(long d) const {
    return impl_->Size()[d];
}

template <typename Real>
const std::vector<long> &tensorfact::TtTensor<Real>::Rank() const {
    return impl_->Rank();
}

template <typename Real>
const long &tensorfact::TtTensor<Real>::Rank(long d) const {
    return impl_->Rank()[d];
}

template <typename Real>
const std::vector<Real> &tensorfact::TtTensor<Real>::Param() const {
    return impl_->Param();
}

template <typename Real>
std::vector<Real> &tensorfact::TtTensor<Real>::Param() {
    return impl_->Param();
}

template <typename Real>
const Real &tensorfact::TtTensor<Real>::Param(long i, long j, long k,
                                              long d) const {
    return impl_->Param(i, j, k, d);
}

template <typename Real>
Real &tensorfact::TtTensor<Real>::Param(long i, long j, long k, long d) {
    return impl_->Param(i, j, k, d);
}

template <typename Real>
const long &tensorfact::TtTensor<Real>::NumParam() const {
    return impl_->NumParam();
}

template <typename Real>
long tensorfact::TtTensor<Real>::NumElement() const {
    return impl_->NumElement();
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator+=(
    const tensorfact::TtTensor<Real> &other) {
    *impl_ += *(other.impl_);
    return *this;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator-=(
    const tensorfact::TtTensor<Real> &other) {
    *impl_ -= *(other.impl_);
    return *this;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator*=(Real alpha) {
    *impl_ *= alpha;
    return *this;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator/=(Real alpha) {
    *impl_ /= alpha;
    return *this;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator*=(
    const tensorfact::TtTensor<Real> &other) {
    *impl_ *= *(other.impl_);
    return *this;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator+(
    const tensorfact::TtTensor<Real> &other) const {
    tensorfact::TtTensor<Real> tt_tensor;
    *(tt_tensor.impl_) = *impl_ + *(other.impl_);
    return tt_tensor;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator-(
    const tensorfact::TtTensor<Real> &other) const {
    tensorfact::TtTensor<Real> tt_tensor;
    *(tt_tensor.impl_) = *impl_ - *(other.impl_);
    return tt_tensor;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator*(
    Real alpha) const {
    tensorfact::TtTensor<Real> tt_tensor;
    *(tt_tensor.impl_) = *impl_ * alpha;
    return tt_tensor;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator/(
    Real alpha) const {
    tensorfact::TtTensor<Real> tt_tensor;
    *(tt_tensor.impl_) = *impl_ / alpha;
    return tt_tensor;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::operator*(
    const tensorfact::TtTensor<Real> &other) const {
    tensorfact::TtTensor<Real> tt_tensor;
    *(tt_tensor.impl_) = *impl_ * *(other.impl_);
    return tt_tensor;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::Concatenate(
    const tensorfact::TtTensor<Real> &other, long d) const {
    tensorfact::TtTensor<Real> tt_tensor;
    *(tt_tensor.impl_) = impl_->Concatenate(*(other.impl_), d);
    return tt_tensor;
}

template <typename Real>
tensorfact::TtTensor<Real> tensorfact::TtTensor<Real>::Shift(long d,
                                                             long shift) const {
    tensorfact::TtTensor<Real> tt_tensor;
    *(tt_tensor.impl_) = impl_->Shift(d, shift);
    return tt_tensor;
}

template <typename Real>
Real tensorfact::TtTensor<Real>::Contract(
    const std::vector<std::vector<Real>> &vectors) const {
    return impl_->Contract(vectors);
}

template <typename Real>
Real tensorfact::TtTensor<Real>::Dot(
    const tensorfact::TtTensor<Real> &other) const {
    return impl_->Dot(*(other.impl_));
}

template <typename Real>
Real tensorfact::TtTensor<Real>::FrobeniusNorm() const {
    return impl_->FrobeniusNorm();
}

template <typename Real>
void tensorfact::TtTensor<Real>::Round(Real relative_tolerance) {
    impl_->Round(relative_tolerance);
}

template <typename Real>
std::vector<Real> tensorfact::TtTensor<Real>::Full() const {
    return impl_->Full();
}

template <typename Real>
Real tensorfact::TtTensor<Real>::Entry(const std::vector<long> &index) const {
    return impl_->Entry(index);
}

template <typename Real>
void tensorfact::TtTensor<Real>::WriteToFile(
    const std::string &file_name) const {
    impl_->WriteToFile(file_name);
}

template <typename Real>
void tensorfact::TtTensor<Real>::ReadFromFile(const std::string &file_name) {
    impl_->ReadFromFile(file_name);
}

// explicit instantiations

template class tensorfact::TtTensor<float>;
template class tensorfact::TtTensor<double>;
