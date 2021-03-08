#include "tensorfact/cp_tensor.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

template <typename Real>
tensorfact::CpTensor<Real>::CpTensor(const arma::field<arma::Mat<Real>> &factor)
    : ndim_(factor.n_elem),
      size_(factor.n_elem),
      rank_(factor(0).n_cols),
      factor_(factor) {
    for (arma::uword d = 0; d < ndim_; ++d) {
        if (d != 0 && factor(d).n_cols != rank_) {
            throw std::logic_error(
                "tensorfact::CpTensor::CpTensor() - Factor matrices must have "
                "the same number of columns");
        }

        size_(d) = factor(d).n_rows;
    }
}

template <typename Real>
Real tensorfact::CpTensor<Real>::operator()(
    const arma::Col<arma::uword> &index) const {
    for (arma::uword d = 0; d < ndim_; ++d) {
        if (index(d) >= size_(d)) {
            throw std::logic_error(
                "tensorfact::CpTensor::operator() - Index out of range");
        }
    }

    Real value = static_cast<Real>(0);

    for (arma::uword r = 0; r < rank_; ++r) {
        Real temp = static_cast<Real>(1);

        for (arma::uword d = 0; d < ndim_; ++d) {
            temp *= factor_(d)(index(d), r);
        }

        value += temp;
    }

    return value;
}

template <typename Real>
void tensorfact::CpTensor<Real>::WriteToFile(
    const std::string &file_name) const {
    std::ofstream file(file_name);

    file << "CP Tensor" << std::endl;

    file << ndim_ << std::endl;

    for (arma::uword d = 0; d < ndim_; ++d) {
        file << size_(d) << std::endl;
    }

    file << rank_ << std::endl;

    file << std::scientific;

    for (arma::uword d = 0; d < ndim_; ++d) {
        for (arma::uword r = 0; r < rank_; ++r) {
            for (arma::uword i = 0; i < size_(d); ++i) {
                file << std::setprecision(17) << factor_(d)(i, r) << std::endl;
            }
        }
    }
}

template <typename Real>
void tensorfact::CpTensor<Real>::ReadFromFile(const std::string &file_name) {
    std::ifstream file(file_name);

    {
        std::string line;
        std::getline(file, line);
        if (line.compare("CP Tensor") != 0) {
            throw std::runtime_error(
                "tensorfact::CpTensor::ReadFromFile() - File does not seem to "
                "contain a CP Tensor");
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

    {
        std::string line;
        std::getline(file, line);
        std::istringstream line_stream(line);
        line_stream >> rank_;
    }

    factor_.set_size(ndim_);
    for (arma::uword d = 0; d < ndim_; ++d) {
        factor_(d).set_size(size_(d), rank_);

        for (arma::uword r = 0; r < rank_; ++r) {
            for (arma::uword i = 0; i < size_(d); ++i) {
                std::string line;
                std::getline(file, line);
                std::istringstream line_stream(line);
                line_stream >> factor_(d)(i, r);
            }
        }
    }
}

// explicit instantiations -----------------------------------------------------

template class tensorfact::CpTensor<float>;
template class tensorfact::CpTensor<double>;
