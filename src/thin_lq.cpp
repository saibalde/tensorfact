#include "thin_lq.hpp"

#include <armadillo>
#include <stdexcept>

template <class Real>
void ThinLq(const arma::Mat<Real> &A, arma::Mat<Real> &L, arma::Mat<Real> &Q) {
    // A = Q * R  =>  A^T = R^T * Q^T

    arma::Mat<Real> L_temp;
    arma::Mat<Real> Q_temp;
    bool status = arma::qr_econ(Q_temp, L_temp, A.t());

    if (!status) {
        throw std::runtime_error("RQ decomposition failed");
    }

    Q = Q_temp.t();
    L = L_temp.t();
}

template void ThinLq<float>(const arma::Mat<float> &A, arma::Mat<float> &L,
                            arma::Mat<float> &Q);
template void ThinLq<double>(const arma::Mat<double> &A, arma::Mat<double> &L,
                             arma::Mat<double> &Q);
