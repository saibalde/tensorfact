#include "Tensorfact_TtTensor.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

template <typename Scalar>
TensorFact::TtTensor<Scalar>::TtTensor(
    const std::vector<TensorFact::Array<Scalar>> &core)
    : ndim_(core.size()),
      size_(core.size()),
      rank_(core.size() + 1),
      core_(core) {
    for (std::size_t d = 0; d < ndim_; ++d) {
        if (core[d].NDim() != 3) {
            throw std::invalid_argument("All cores must be 3D arrays");
        }
    }

    for (std::size_t d = 0; d < ndim_; ++d) {
        rank_[d] = core[d].Size(0);
        size_[d] = core[d].Size(1);

        if ((d > 0) && (core[d - 1].Size(2) != rank_[d])) {
            throw std::invalid_argument(
                "Core sizes are incompatible with the TT format");
        }
    }
    rank_[ndim_] = core[ndim_ - 1].Size(2);

    if ((rank_[0] != 1) || (rank_[ndim_] != 1)) {
        throw std::invalid_argument("Boundary ranks must be one");
    }
}

template <typename Scalar>
Scalar TensorFact::TtTensor<Scalar>::operator()(
    const std::vector<std::size_t> &index) const {
    TensorFact::Array<Scalar> temp;

    for (std::size_t d = 0; d < ndim_; ++d) {
        TensorFact::Array<Scalar> slice;
        slice.Resize({rank_[d], rank_[d + 1]});

        for (std::size_t j = 0; j < rank_[d + 1]; ++j) {
            for (std::size_t i = 0; i < rank_[d]; ++i) {
                slice({i, j}) = core_[d]({i, index[d], j});
            }
        }

        if (d == 0) {
            temp = slice;
        } else {
            TensorFact::Array<Scalar> temp2;
            temp.Multiply(false, slice, false, temp2);
            temp = temp2;
        }
    }

    return temp({0, 0});
}

template <typename Scalar>
void TensorFact::TtTensor<Scalar>::WriteToFile(
    const std::string &file_name) const {
    std::ofstream file(file_name);

    file << "TT Tensor" << std::endl;

    file << ndim_ << std::endl;

    for (std::size_t d = 0; d < ndim_; ++d) {
        file << size_[d] << std::endl;
    }

    for (std::size_t d = 0; d <= ndim_; ++d) {
        file << rank_[d] << std::endl;
    }

    file << std::scientific;

    for (std::size_t d = 0; d < ndim_; ++d) {
        const std::size_t num_element = core_[d].NumberOfElements();
        for (std::size_t n = 0; n < num_element; ++n) {
            file << std::setprecision(17) << core_[d](n) << std::endl;
        }
    }
}

template <typename Scalar>
void TensorFact::TtTensor<Scalar>::ReadFromFile(const std::string &file_name) {
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
    for (std::size_t d = 0; d < ndim_; ++d) {
        std::string line;
        std::getline(file, line);
        std::istringstream line_stream(line);
        line_stream >> size_[d];
    }

    rank_.resize(ndim_ + 1);
    for (std::size_t d = 0; d <= ndim_; ++d) {
        std::string line;
        std::getline(file, line);
        std::istringstream line_stream(line);
        line_stream >> rank_[d];
    }

    core_.resize(ndim_);
    for (std::size_t d = 0; d < ndim_; ++d) {
        core_[d].Resize({rank_[d], size_[d], rank_[d + 1]});
        const std::size_t num_element = core_[d].NumberOfElements();
        for (std::size_t n = 0; n < num_element; ++n) {
            std::string line;
            std::getline(file, line);
            std::istringstream line_stream(line);
            line_stream >> core_[d](n);
        }
    }
}

template <typename Scalar>
TensorFact::TtTensor<Scalar> TensorFact::TtTensor<Scalar>::operator+(
    const TtTensor<Scalar> &other) const {
    if (ndim_ != other.ndim_) {
        throw std::invalid_argument("TT tensors must have same dimensionality");
    }

    for (std::size_t d = 0; d < ndim_; ++d) {
        if (size_[d] != other.size_[d]) {
            throw std::invalid_argument("TT tensors must have same size");
        }
    }

    // new ranks
    std::vector<std::size_t> rank_new(ndim_ + 1);
    rank_new[0] = 1;
    for (std::size_t d = 1; d < ndim_; ++d) {
        rank_new[d] = rank_[d] + other.rank_[d];
    }
    rank_new[ndim_] = 1;

    // new cores
    std::vector<TensorFact::Array<Scalar>> core_new(ndim_);

    // first core
    core_new[0].Resize({1, size_[0], rank_new[1]});

    for (std::size_t k = 0; k < rank_[1]; ++k) {
        for (std::size_t j = 0; j < size_[0]; ++j) {
            core_new[0]({0, j, k}) = core_[0]({0, j, k});
        }
    }

    for (std::size_t k = 0; k < other.rank_[1]; ++k) {
        for (std::size_t j = 0; j < size_[0]; ++j) {
            core_new[0]({0, j, rank_[1] + k}) = other.core_[0]({0, j, k});
        }
    }

    // middle cores
    for (std::size_t d = 1; d < ndim_ - 1; ++d) {
        core_new[d].Resize({rank_new[d], size_[d], rank_new[d + 1]});

        for (std::size_t k = 0; k < rank_[d + 1]; ++k) {
            for (std::size_t j = 0; j < size_[d]; ++j) {
                for (std::size_t i = 0; i < rank_[d]; ++i) {
                    core_new[d]({i, j, k}) = core_[d]({i, j, k});
                }
            }
        }

        for (std::size_t k = 0; k < other.rank_[d + 1]; ++k) {
            for (std::size_t j = 0; j < size_[d]; ++j) {
                for (std::size_t i = 0; i < other.rank_[d]; ++i) {
                    core_new[d]({rank_[d] + i, j, rank_[d + 1] + k}) =
                        other.core_[d]({i, j, k});
                }
            }
        }
    }

    // last core
    core_new[ndim_ - 1].Resize({rank_new[ndim_ - 1], size_[ndim_ - 1], 1});

    for (std::size_t j = 0; j < size_[ndim_ - 1]; ++j) {
        for (std::size_t i = 0; i < rank_[ndim_ - 1]; ++i) {
            core_new[ndim_ - 1]({i, j, 0}) = core_[ndim_ - 1]({i, j, 0});
        }

        for (std::size_t i = 0; i < other.rank_[ndim_ - 1]; ++i) {
            core_new[ndim_ - 1]({rank_[ndim_ - 1] + i, j, 0}) =
                other.core_[ndim_ - 1]({i, j, 0});
        }
    }

    return TtTensor<Scalar>(core_new);
}

template <typename Scalar>
TensorFact::TtTensor<Scalar> TensorFact::TtTensor<Scalar>::operator*(
    Scalar alpha) const {
    std::vector<TensorFact::Array<Scalar>> core_new = core_;

    core_new[0] = alpha * core_new[0];

    return TtTensor<Scalar>(core_new);
}

template <typename Scalar>
TensorFact::TtTensor<Scalar> TensorFact::TtTensor<Scalar>::operator-(
    const TensorFact::TtTensor<Scalar> &other) const {
    return *this + other * static_cast<Scalar>(-1.0);
}

template <typename Scalar>
TensorFact::TtTensor<Scalar> TensorFact::TtTensor<Scalar>::operator/(
    Scalar alpha) const {
    if (std::abs(alpha) < std::numeric_limits<Scalar>::epsilon()) {
        throw std::logic_error("Dividing by a value too close to zero");
    }

    return *this * (static_cast<Scalar>(1) / alpha);
}

template <typename Scalar>
void TensorFact::TtTensor<Scalar>::ComputeFromFull(
    const TensorFact::Array<Scalar> &array, Scalar rel_acc) {
    if (rel_acc <= std::numeric_limits<Scalar>::epsilon()) {
        throw std::logic_error("Required accuracy is too small");
    }

    ndim_ = array.NDim();
    size_ = array.Size();
    rank_ = std::vector<std::size_t>(ndim_ + 1);
    core_ = std::vector<TensorFact::Array<Scalar>>(ndim_);

    const Scalar delta = rel_acc / std::sqrt(ndim_ - 1);

    core_[ndim_ - 1] = array;

    core_[ndim_ - 1].Reshape(
        {size_[0], core_[ndim_ - 1].NumberOfElements() / size_[0]});
    rank_[0] = 1;
    rank_[ndim_] = 1;

    for (std::size_t d = 0; d < ndim_ - 1; ++d) {
        TensorFact::Array<Scalar> s;
        TensorFact::Array<Scalar> Vt;
        core_[ndim_ - 1].TruncatedSvd(core_[d], s, Vt, delta, true);

        rank_[d + 1] = s.NumberOfElements();

        core_[d].Reshape({rank_[d], size_[d], rank_[d + 1]});

        core_[ndim_ - 1].Resize(Vt.Size());
        for (std::size_t j = 0; j < Vt.Size(1); ++j) {
            for (std::size_t i = 0; i < Vt.Size(0); ++i) {
                core_[ndim_ - 1]({i, j}) = s({i}) * Vt({i, j});
            }
        }

        core_[ndim_ - 1].Reshape(
            {rank_[d + 1] * size_[d + 1], core_[ndim_ - 1].NumberOfElements() /
                                              (rank_[d + 1] * size_[d + 1])});
    }

    core_[ndim_ - 1].Reshape(
        {rank_[ndim_ - 1], size_[ndim_ - 1], rank_[ndim_]});
}

template <typename Scalar>
void TensorFact::TtTensor<Scalar>::Round(Scalar rel_acc) {
    // right-to-left orthogonalization
    for (std::size_t d = ndim_ - 1; d > 0; --d) {
        core_[d].Reshape({rank_[d], size_[d] * rank_[d + 1]});

        TensorFact::Array<Scalar> R;
        TensorFact::Array<Scalar> Q;
        core_[d].ReducedRq(R, Q);

        std::size_t r = R.Size(1);

        core_[d] = Q;
        core_[d].Reshape({r, size_[d], rank_[d + 1]});

        core_[d - 1].Reshape({rank_[d - 1] * size_[d - 1], rank_[d]});

        TensorFact::Array<Scalar> temp;
        core_[d - 1].Multiply(false, R, false, temp);

        core_[d - 1] = temp;
        core_[d - 1].Reshape({rank_[d - 1], size_[d - 1], r});

        rank_[d] = r;
    }

    // left-to-right compression
    if (rel_acc > std::numeric_limits<Scalar>::epsilon()) {
        const Scalar delta = rel_acc / std::sqrt(ndim_ - 1);

        for (std::size_t d = 0; d < ndim_ - 1; ++d) {
            core_[d].Reshape({rank_[d] * size_[d], rank_[d + 1]});

            TensorFact::Array<Scalar> U;
            TensorFact::Array<Scalar> s;
            TensorFact::Array<Scalar> Vt;
            core_[d].TruncatedSvd(U, s, Vt, delta, true);

            const std::size_t r = s.NumberOfElements();

            core_[d] = U;
            core_[d].Reshape({rank_[d], size_[d], r});

            core_[d + 1].Reshape({rank_[d + 1], size_[d + 1] * rank_[d + 2]});

            for (std::size_t j = 0; j < Vt.Size(1); ++j) {
                for (std::size_t i = 0; i < Vt.Size(0); ++i) {
                    Vt({i, j}) *= s({i});
                }
            }

            TensorFact::Array<Scalar> temp;
            Vt.Multiply(false, core_[d + 1], false, temp);

            core_[d + 1] = temp;
            core_[d + 1].Reshape({r, size_[d + 1], rank_[d + 2]});

            rank_[d + 1] = r;
        }
    }
}

template <typename Scalar>
TensorFact::TtTensor<Scalar> TensorFact::TtTensor<Scalar>::Concatenate(
    const TensorFact::TtTensor<Scalar> &other, std::size_t dim,
    Scalar rel_acc) const {
    if (ndim_ != other.ndim_) {
        throw std::invalid_argument(
            "Dimensionality of the two tensors must be the same");
    }

    if (dim >= ndim_) {
        throw std::invalid_argument("Invalid concatenation dimension");
    }

    for (std::size_t d = 0; d < ndim_; ++d) {
        if (d != dim && size_[d] != other.size_[d]) {
            throw std::invalid_argument(
                "Tensor sizes must match apart from the concatenation "
                "dimension");
        }
    }

    const std::size_t size_1 = size_[dim];
    const std::size_t size_2 = other.size_[dim];

    const TensorFact::TtTensor<Scalar> tensor_1 =
        this->AddZeroPaddingBack(dim, size_2);
    const TensorFact::TtTensor<Scalar> tensor_2 =
        other.AddZeroPaddingFront(dim, size_1);

    TensorFact::TtTensor<Scalar> tensor = tensor_1 + tensor_2;
    tensor.Round(rel_acc);

    return tensor;
}

template <typename Scalar>
Scalar TensorFact::TtTensor<Scalar>::Dot(
    const TensorFact::TtTensor<Scalar> &other) const {
    if (ndim_ != other.ndim_) {
        throw std::invalid_argument(
            "Two tensors must have the same dimensionality");
    }

    for (std::size_t d = 0; d < ndim_; ++d) {
        if (size_[d] != other.size_[d]) {
            throw std::invalid_argument("Two tensors must have the same size");
        }
    }

    TensorFact::Array<Scalar> temp1;

    for (std::size_t l = 0; l < ndim_; ++l) {
        std::size_t d = ndim_ - 1 - l;

        TensorFact::Array<Scalar> slice;
        slice.Resize({rank_[d], rank_[d + 1]});

        TensorFact::Array<Scalar> other_slice;
        other_slice.Resize({other.rank_[d], other.rank_[d + 1]});

        std::vector<TensorFact::Array<Scalar>> temp2(size_[d]);

        if (d == ndim_ - 1) {
            // Kronecker product of the last cores
            for (std::size_t k = 0; k < size_[d]; ++k) {
                for (std::size_t j = 0; j < rank_[d + 1]; ++j) {
                    for (std::size_t i = 0; i < rank_[d]; ++i) {
                        slice({i, j}) = core_[d]({i, k, j});
                    }
                }

                for (std::size_t j = 0; j < other.rank_[d + 1]; ++j) {
                    for (std::size_t i = 0; i < other.rank_[d]; ++i) {
                        other_slice({i, j}) = other.core_[d]({i, k, j});
                    }
                }

                other_slice.Multiply(false, slice, true, temp2[k]);
            }
        } else {
            // multiplication by Kronecker product of the cores
            for (std::size_t k = 0; k < size_[d]; ++k) {
                for (std::size_t j = 0; j < rank_[d + 1]; ++j) {
                    for (std::size_t i = 0; i < rank_[d]; ++i) {
                        slice({i, j}) = core_[d]({i, k, j});
                    }
                }

                for (std::size_t j = 0; j < other.rank_[d + 1]; ++j) {
                    for (std::size_t i = 0; i < other.rank_[d]; ++i) {
                        other_slice({i, j}) = other.core_[d]({i, k, j});
                    }
                }

                TensorFact::Array<Scalar> temp3;
                other_slice.Multiply(false, temp1, false, temp3);

                temp3.Multiply(false, slice, true, temp2[k]);
            }
        }

        temp1.Resize({other.rank_[d], rank_[d]});
        for (std::size_t j = 0; j < rank_[d]; ++j) {
            for (std::size_t i = 0; i < other.rank_[d]; ++i) {
                Scalar sum = static_cast<Scalar>(0);
                for (std::size_t k = 0; k < size_[d]; ++k) {
                    sum += temp2[k]({i, j});
                }

                temp1({i, j}) = sum;
            }
        }
    }

    return temp1({0, 0});
}

template <typename Scalar>
TensorFact::TtTensor<Scalar> TensorFact::TtTensor<Scalar>::AddZeroPaddingBack(
    std::size_t dim, std::size_t pad) const {
    std::vector<TensorFact::Array<Scalar>> core(ndim_);

    for (std::size_t d = 0; d < ndim_; ++d) {
        if (d != dim) {
            core[d] = core_[d];
        }
    }

    core[dim].Resize({rank_[dim], size_[dim] + pad, rank_[dim + 1]});
    for (std::size_t k = 0; k < rank_[dim + 1]; ++k) {
        for (std::size_t j = 0; j < size_[dim]; ++j) {
            for (std::size_t i = 0; i < rank_[dim]; ++i) {
                core[dim]({i, j, k}) = core_[dim]({i, j, k});
            }
        }

        for (std::size_t j = 0; j < pad; ++j) {
            for (std::size_t i = 0; i < rank_[dim]; ++i) {
                core[dim]({i, j + size_[dim], k}) = static_cast<Scalar>(0);
            }
        }
    }

    return TensorFact::TtTensor<Scalar>(core);
}

template <typename Scalar>
TensorFact::TtTensor<Scalar> TensorFact::TtTensor<Scalar>::AddZeroPaddingFront(
    std::size_t dim, std::size_t pad) const {
    std::vector<TensorFact::Array<Scalar>> core(ndim_);

    for (std::size_t d = 0; d < ndim_; ++d) {
        if (d != dim) {
            core[d] = core_[d];
        }
    }

    core[dim].Resize({rank_[dim], size_[dim] + pad, rank_[dim + 1]});
    for (std::size_t k = 0; k < rank_[dim + 1]; ++k) {
        for (std::size_t j = 0; j < pad; ++j) {
            for (std::size_t i = 0; i < rank_[dim]; ++i) {
                core[dim]({i, j, k}) = static_cast<Scalar>(0);
            }
        }

        for (std::size_t j = 0; j < size_[dim]; ++j) {
            for (std::size_t i = 0; i < rank_[dim]; ++i) {
                core[dim]({i, j + pad, k}) = core_[dim]({i, j, k});
            }
        }
    }

    return TensorFact::TtTensor<Scalar>(core);
}

// explicit instantiations -----------------------------------------------------

template class TensorFact::TtTensor<float>;
template class TensorFact::TtTensor<double>;
