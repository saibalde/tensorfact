#include "TensorFact_CpTensor.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

template <typename Scalar>
TensorFact::CpTensor<Scalar>::CpTensor(
    const std::vector<TensorFact::Array<Scalar>> &factor)
    : ndim_(factor.size()), size_(factor.size()), rank_(0), factor_(factor) {
    for (std::size_t d = 0; d < ndim_; ++d) {
        if (factor_[d].NDim() != 2) {
            throw std::invalid_argument(
                "Factors must be a vector of 2D arrays");
        }

        if (d == 0) {
            rank_ = factor[0].Size(1);
        } else if (factor_[d].Size(1) != rank_) {
            throw std::invalid_argument(
                "Factors must have the same number of columns");
        }

        size_[d] = factor[d].Size(0);
    }
}

template <typename Scalar>
Scalar TensorFact::CpTensor<Scalar>::operator()(
    const std::vector<std::size_t> &index) const {
    if (index.size() != ndim_) {
        throw std::invalid_argument(
            "Size of index array does not match dimensionality");
    }

    for (std::size_t d = 0; d < ndim_; ++d) {
        if (index[d] >= size_[d]) {
            throw std::invalid_argument("Index is out of range");
        }
    }

    Scalar value = static_cast<Scalar>(0);

    for (std::size_t r = 0; r < rank_; ++r) {
        Scalar temp = static_cast<Scalar>(1);

        for (std::size_t d = 0; d < ndim_; ++d) {
            temp *= factor_[d]({index[d], r});
        }

        value += temp;
    }

    return value;
}

template <typename Scalar>
void TensorFact::CpTensor<Scalar>::WriteToFile(
    const std::string &file_name) const {
    std::ofstream file(file_name);

    file << "CP Tensor" << std::endl;

    file << ndim_ << std::endl;

    for (std::size_t d = 0; d < ndim_; ++d) {
        file << size_[d] << std::endl;
    }

    file << rank_ << std::endl;

    file << std::scientific;

    for (std::size_t d = 0; d < ndim_; ++d) {
        for (std::size_t r = 0; r < rank_; ++r) {
            for (std::size_t i = 0; i < size_[d]; ++i) {
                file << std::setprecision(17) << factor_[d]({i, r})
                     << std::endl;
            }
        }
    }
}

template <typename Scalar>
void TensorFact::CpTensor<Scalar>::ReadFromFile(const std::string &file_name) {
    std::ifstream file(file_name);

    {
        std::string line;
        std::getline(file, line);
        if (line.compare("CP Tensor") != 0) {
            throw std::runtime_error("File is missing CP tensor header");
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

    {
        std::string line;
        std::getline(file, line);
        std::istringstream line_stream(line);
        line_stream >> rank_;
    }

    factor_.resize(ndim_);
    for (std::size_t d = 0; d < ndim_; ++d) {
        factor_[d].Resize({size_[d], rank_});

        for (std::size_t r = 0; r < rank_; ++r) {
            for (std::size_t i = 0; i < size_[d]; ++i) {
                std::string line;
                std::getline(file, line);
                std::istringstream line_stream(line);
                line_stream >> factor_[d]({i, r});
            }
        }
    }
}

// explicit instantiations -----------------------------------------------------

template class TensorFact::CpTensor<float>;
template class TensorFact::CpTensor<double>;
