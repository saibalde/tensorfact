#include "svd.hpp"

#include <stdexcept>

template <typename Real>
void tensorfact::TruncatedSvd(const arma::Mat<Real> &A, Real delta,
                              arma::Mat<Real> &U, arma::Col<Real> &s,
                              arma::Mat<Real> &V, arma::uword &rank) {
    arma::Mat<Real> U_full;
    arma::Col<Real> s_full;
    arma::Mat<Real> V_full;

    bool status = arma::svd(U_full, s_full, V_full, A);
    if (!status) {
        throw std::runtime_error("TruncatedSvd() - Could not compute SVD");
    }

    rank = s_full.n_elem;
    Real residue = 0.0;

    while (true) {
        residue += std::pow(s_full(rank - 1), 2);
        if (residue > delta) {
            break;
        }
        rank -= 1;
    }

    U = U_full.cols(0, rank - 1);
    s = s_full(arma::span(0, rank - 1));
    V = V_full.cols(0, rank - 1);
}

// explicit instantiation ------------------------------------------------------

template void tensorfact::TruncatedSvd<float>(const arma::Mat<float> &, float,
                                              arma::Mat<float> &,
                                              arma::Col<float> &,
                                              arma::Mat<float> &,
                                              arma::uword &);
template void tensorfact::TruncatedSvd<double>(const arma::Mat<double> &,
                                               double, arma::Mat<double> &,
                                               arma::Col<double> &,
                                               arma::Mat<double> &,
                                               arma::uword &);
