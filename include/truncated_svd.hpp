#ifndef TRUNCATED_SVD_HPP
#define TRUNCATED_SVD_HPP

#include <armadillo>
#include <stdexcept>

/// Truncated SVD given error tolerance
template <typename Real, typename Index>
void truncatedSvd(const arma::Mat<Real> &A, Real deltaSquare,
                  arma::Mat<Real> &U, arma::Col<Real> &s, arma::Mat<Real> &V,
                  Index &rank) {
    arma::Mat<Real> UFull;
    arma::Col<Real> sFull;
    arma::Mat<Real> VFull;

    bool status = arma::svd(UFull, sFull, VFull, A);
    if (!status) {
        throw std::runtime_error("truncatedSvd() - Could not compute SVD");
    }

    rank = sFull.n_elem;
    Real residue = 0.0;
    while (true) {
        residue += std::pow(sFull(rank - 1), 2);
        if (residue > deltaSquare) {
            break;
        }
        rank -= 1;
    }

    U = UFull.cols(0, rank - 1);
    s = sFull(arma::span(0, rank - 1));
    V = VFull.cols(0, rank - 1);
}

#endif
