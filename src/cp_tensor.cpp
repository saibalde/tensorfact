#include "tensorfact/cp_tensor.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

tensorfact::CpTensor::CpTensor(int ndim, const std::vector<int> &size, int rank,
                               const std::vector<double> &param)
    : ndim_(ndim), size_(size), rank_(rank), offset_(ndim + 1), param_(param) {
    // validate inputs
    if (ndim_ < 1) {
        throw std::invalid_argument("Dimension must be at least one");
    }

    if (size_.size() != ndim_) {
        throw std::invalid_argument(
            "Length of the size vector is incompatible with specified "
            "dimension");
    }

    for (int d = 0; d < ndim_; ++d) {
        if (size_[d] < 1) {
            throw std::invalid_argument("Sizes must be at least one");
        }
    }

    if (rank_ < 1) {
        throw std::invalid_argument("Rank must be at least one");
    }

    offset_[0] = 0;
    for (int d = 0; d < ndim_; ++d) {
        offset_[d + 1] = offset_[d] + size_[d] * rank_;
    }

    if (param_.size() != offset_[ndim_]) {
        throw std::invalid_argument(
            "Number of entries in the parameter vector does not match size and "
            "rank specification");
    }
}

double tensorfact::CpTensor::Entry(const std::vector<int> &index) const {
    if (index.size() != ndim_) {
        throw std::invalid_argument(
            "Size of index array does not match dimensionality");
    }

    for (int d = 0; d < ndim_; ++d) {
        if (index[d] < 0 || index[d] >= size_[d]) {
            throw std::out_of_range("Index is out of range");
        }
    }

    double value = 0.0;

    for (int r = 0; r < rank_; ++r) {
        double temp = 1.0;

        for (int d = 0; d < ndim_; ++d) {
            temp *= param_[LinearIndex(index[d], r, d)];
        }

        value += temp;
    }

    return value;
}

void tensorfact::CpTensor::WriteToFile(const std::string &file_name) const {
    std::ofstream file(file_name);

    file << "CP Tensor" << std::endl;

    file << ndim_ << std::endl;

    for (int d = 0; d < ndim_; ++d) {
        file << size_[d] << std::endl;
    }

    file << rank_ << std::endl;

    file << std::scientific;

    for (int d = 0; d < ndim_; ++d) {
        for (int r = 0; r < rank_; ++r) {
            for (int i = 0; i < size_[d]; ++i) {
                file << std::setprecision(17) << param_[LinearIndex(i, r, d)]
                     << std::endl;
            }
        }
    }

    file << std::defaultfloat;
}

void tensorfact::CpTensor::ReadFromFile(const std::string &file_name) {
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
    for (int d = 0; d < ndim_; ++d) {
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

    offset_.resize(ndim_ + 1);
    offset_[0] = 0;
    for (int d = 0; d < ndim_; ++d) {
        offset_[d + 1] = offset_[d] + size_[d] * rank_;
    }

    param_.resize(offset_[ndim_]);
    for (int d = 0; d < ndim_; ++d) {
        for (int r = 0; r < rank_; ++r) {
            for (int i = 0; i < size_[d]; ++i) {
                std::string line;
                std::getline(file, line);
                std::istringstream line_stream(line);
                line_stream >> param_[LinearIndex(i, r, d)];
            }
        }
    }
}

int tensorfact::CpTensor::LinearIndex(int i, int r, int d) const {
    return offset_[d] + i + size_[d] * r;
}
