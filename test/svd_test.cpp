#include "svd.hpp"

#include <armadillo>
#include <vector>

#include "gtest/gtest.h"

TEST(svd, TruncatedSvd_float) {
    const arma::uword m = 4;
    const arma::uword n = 3;

    arma::Mat<float> X(4, 3, arma::fill::zeros);
    X(0, 0) = 0.8730758679812514;
    X(1, 0) = 0.8905818857769324;
    X(2, 0) = 1.0004504376390704;
    X(3, 0) = 1.2121943608516716;
    X(0, 1) = 1.039764271321522;
    X(1, 1) = 0.9910014553992643;
    X(2, 1) = 0.7262655727821424;
    X(3, 1) = 0.9997665065853555;
    X(0, 2) = 0.5079667994149603;
    X(1, 2) = 0.6486914410114907;
    X(2, 2) = 1.1665137875687934;
    X(3, 2) = 1.023738282971006;

    arma::Mat<float> U_true(4, 3, arma::fill::zeros);
    U_true(0, 0) = -0.4374561669866752;
    U_true(1, 0) = -0.4552544328197976;
    U_true(2, 0) = -0.514262977748206;
    U_true(3, 0) = -0.5804387074280961;
    U_true(0, 1) = -0.6166991646254513;
    U_true(1, 1) = -0.361342760559755;
    U_true(2, 1) = 0.6849330008504085;
    U_true(3, 1) = 0.14135180963603095;
    U_true(0, 2) = -0.10536900266635602;
    U_true(1, 2) = -0.3861054867694944;
    U_true(2, 2) = -0.4619084065205092;
    U_true(3, 2) = 0.7914926091564425;

    arma::Col<float> s_true(3, arma::fill::zeros);
    s_true(0) = 3.2265338169944235;
    s_true(1) = 0.5355825867628933;
    s_true(2) = 0.07847516615102999;

    arma::Mat<float> V_true(3, 3, arma::fill::zeros);
    V_true(0, 0) = -0.6215564555795905;
    V_true(1, 0) = -0.5764091990342418;
    V_true(2, 0) = -0.5304903465625369;
    V_true(0, 1) = -0.0068021041920685536;
    V_true(1, 1) = -0.673193835938418;
    V_true(2, 1) = 0.7394347778087524;
    V_true(0, 2) = 0.7833398393455403;
    V_true(1, 2) = -0.46320891023729516;
    V_true(2, 2) = -0.4145071791548137;

    const float delta = 0.25;

    arma::Mat<float> U;
    arma::Col<float> s;
    arma::Mat<float> V;
    arma::uword r;
    tensorfact::TruncatedSvd<float>(X, delta, U, s, V, r);

    ASSERT_TRUE(r == 2);

    for (arma::uword j = 0; j < r; ++j) {
        for (arma::uword i = 0; i < m; ++i) {
            ASSERT_TRUE(std::abs(U(i, j) - U_true(i, j)) < 1.0e-06);
        }
    }

    for (arma::uword i = 0; i < r; ++i) {
        ASSERT_TRUE(std::abs(s(i) - s_true(i)) < 1.0e-06);
    }

    for (arma::uword j = 0; j < r; ++j) {
        for (arma::uword i = 0; i < n; ++i) {
            ASSERT_TRUE(std::abs(V(i, j) - V_true(i, j)) < 1.0e-06);
        }
    }
}

TEST(svd, TruncatedSvd_double) {
    const arma::uword m = 4;
    const arma::uword n = 3;

    arma::Mat<double> X(4, 3, arma::fill::zeros);
    X(0, 0) = 0.8730758679812514;
    X(1, 0) = 0.8905818857769324;
    X(2, 0) = 1.0004504376390704;
    X(3, 0) = 1.2121943608516716;
    X(0, 1) = 1.039764271321522;
    X(1, 1) = 0.9910014553992643;
    X(2, 1) = 0.7262655727821424;
    X(3, 1) = 0.9997665065853555;
    X(0, 2) = 0.5079667994149603;
    X(1, 2) = 0.6486914410114907;
    X(2, 2) = 1.1665137875687934;
    X(3, 2) = 1.023738282971006;

    arma::Mat<double> U_true(4, 3, arma::fill::zeros);
    U_true(0, 0) = -0.4374561669866752;
    U_true(1, 0) = -0.4552544328197976;
    U_true(2, 0) = -0.514262977748206;
    U_true(3, 0) = -0.5804387074280961;
    U_true(0, 1) = 0.6166991646254513;
    U_true(1, 1) = 0.361342760559755;
    U_true(2, 1) = -0.6849330008504085;
    U_true(3, 1) = -0.14135180963603095;
    U_true(0, 2) = 0.10536900266635602;
    U_true(1, 2) = 0.3861054867694944;
    U_true(2, 2) = 0.4619084065205092;
    U_true(3, 2) = -0.7914926091564425;

    arma::Col<double> s_true(3, arma::fill::zeros);
    s_true(0) = 3.2265338169944235;
    s_true(1) = 0.5355825867628933;
    s_true(2) = 0.07847516615102999;

    arma::Mat<double> V_true(3, 3, arma::fill::zeros);
    V_true(0, 0) = -0.6215564555795905;
    V_true(1, 0) = -0.5764091990342418;
    V_true(2, 0) = -0.5304903465625369;
    V_true(0, 1) = 0.0068021041920685536;
    V_true(1, 1) = 0.673193835938418;
    V_true(2, 1) = -0.7394347778087524;
    V_true(0, 2) = -0.7833398393455403;
    V_true(1, 2) = 0.46320891023729516;
    V_true(2, 2) = 0.4145071791548137;

    const double delta = 0.25;

    arma::Mat<double> U;
    arma::Col<double> s;
    arma::Mat<double> V;
    arma::uword r;
    tensorfact::TruncatedSvd<double>(X, delta, U, s, V, r);

    ASSERT_TRUE(r == 2);

    for (arma::uword j = 0; j < r; ++j) {
        for (arma::uword i = 0; i < m; ++i) {
            ASSERT_TRUE(std::abs(U(i, j) - U_true(i, j)) < 1.0e-15);
        }
    }

    for (arma::uword i = 0; i < r; ++i) {
        ASSERT_TRUE(std::abs(s(i) - s_true(i)) < 1.0e-15);
    }

    for (arma::uword j = 0; j < r; ++j) {
        for (arma::uword i = 0; i < n; ++i) {
            ASSERT_TRUE(std::abs(V(i, j) - V_true(i, j)) < 1.0e-15);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
