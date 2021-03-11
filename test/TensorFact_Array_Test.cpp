#include "TensorFact_Array.hpp"

#include <cmath>
#include <random>

#include "gtest/gtest.h"

TEST(Array, ZeroDimensional) {
    TensorFact::Array<float> array;
    array.Resize({});

    ASSERT_TRUE(array.NDim() == 0);

    ASSERT_TRUE(array.NumberOfElements() == 0);
}

TEST(Array, OneDimensional) {
    TensorFact::Array<double> array;
    array.Resize({5});

    ASSERT_TRUE(array.NDim() == 1);

    ASSERT_TRUE(array.Size()[0] == 5);

    ASSERT_TRUE(array.Size(0) == 5);

    ASSERT_TRUE(array.NumberOfElements() == 5);

    for (std::size_t i = 0; i < 5; ++i) {
        array({i}) = static_cast<double>(i);
    }

    for (std::size_t n = 0; n < 5; ++n) {
        ASSERT_TRUE(std::abs(array(n) - n) < 1.0e-15);
    }
}

TEST(Array, TwoDimensional) {
    TensorFact::Array<float> array;
    array.Resize({4, 6});

    ASSERT_TRUE(array.NDim() == 2);

    ASSERT_TRUE(array.Size()[0] == 4);
    ASSERT_TRUE(array.Size()[1] == 6);

    ASSERT_TRUE(array.Size(0) == 4);
    ASSERT_TRUE(array.Size(1) == 6);

    ASSERT_TRUE(array.NumberOfElements() == 24);

    for (std::size_t n = 0; n < 24; ++n) {
        array(n) = static_cast<float>(n);
    }

    for (std::size_t j = 0; j < 6; ++j) {
        for (std::size_t i = 0; i < 4; ++i) {
            ASSERT_TRUE(std::abs(array({i, j}) - (i + 4 * j)) < 1.0e-06);
        }
    }
}

TEST(Array, ThreeDimensional) {
    TensorFact::Array<double> array;
    array.Resize({3, 7, 5});

    ASSERT_TRUE(array.NDim() == 3);

    ASSERT_TRUE(array.Size()[0] == 3);
    ASSERT_TRUE(array.Size()[1] == 7);
    ASSERT_TRUE(array.Size()[2] == 5);

    ASSERT_TRUE(array.Size(0) == 3);
    ASSERT_TRUE(array.Size(1) == 7);
    ASSERT_TRUE(array.Size(2) == 5);

    ASSERT_TRUE(array.NumberOfElements() == 105);

    for (std::size_t k = 0; k < 5; ++k) {
        for (std::size_t j = 0; j < 7; ++j) {
            for (std::size_t i = 0; i < 3; ++i) {
                array({i, j, k}) = static_cast<double>(i + 3 * j + 21 * k);
            }
        }
    }

    for (std::size_t n = 0; n < 105; ++n) {
        ASSERT_TRUE(std::abs(array(n) - n) < 1.0e-15);
    }
}

TEST(Array, Reshape) {
    TensorFact::Array<float> array;
    array.Resize({3, 7, 5});

    for (std::size_t k = 0; k < 5; ++k) {
        for (std::size_t j = 0; j < 7; ++j) {
            for (std::size_t i = 0; i < 3; ++i) {
                array({i, j, k}) = i + 3 * j + 21 * k;
            }
        }
    }

    array.Reshape({7, 3, 5});

    for (std::size_t n = 0; n < 105; ++n) {
        ASSERT_TRUE(std::abs(array(n) - n) < 1.0e-05);
    }

    array.Reshape({15, 7});
    for (std::size_t j = 0; j < 7; ++j) {
        for (std::size_t i = 0; i < 15; ++i) {
            ASSERT_TRUE(std::abs(array({i, j}) - (i + 15 * j)) < 1.0e-05);
        }
    }
}

TEST(Array, MatrixVectorMultiply) {
    TensorFact::Array<double> A;
    A.Resize({4, 5});
    for (std::size_t j = 0; j < 5; ++j) {
        for (std::size_t i = 0; i < 4; ++i) {
            A({i, j}) = static_cast<double>(i + j);
        }
    }

    {
        TensorFact::Array<double> x;
        x.Resize({5});
        for (std::size_t j = 0; j < 5; ++j) {
            x({j}) = static_cast<double>(1);
        }

        TensorFact::Array<double> b;
        b.Resize({4});
        for (std::size_t i = 0; i < 4; ++i) {
            b({i}) = static_cast<double>(0);
            for (std::size_t j = 0; j < 5; ++j) {
                b({i}) += static_cast<double>(i + j);
            }
        }

        {
            TensorFact::Array<double> y;
            A.Multiply(false, x, false, y);

            for (std::size_t i = 0; i < 4; ++i) {
                ASSERT_TRUE(std::abs(y({i}) - b({i})) < 1.0e-15);
            }
        }

        {
            TensorFact::Array<double> y;
            A.Multiply(false, x, true, y);

            for (std::size_t i = 0; i < 4; ++i) {
                ASSERT_TRUE(std::abs(y({i}) - b({i})) < 1.0e-15);
            }
        }
    }

    {
        TensorFact::Array<double> x;
        x.Resize({4});
        for (std::size_t i = 0; i < 4; ++i) {
            x({i}) = static_cast<double>(1);
        }

        TensorFact::Array<double> b;
        b.Resize({5});
        for (std::size_t j = 0; j < 5; ++j) {
            b({j}) = static_cast<double>(0);
            for (std::size_t i = 0; i < 4; ++i) {
                b({j}) += static_cast<double>(i + j);
            }
        }

        {
            TensorFact::Array<double> y;
            A.Multiply(true, x, false, y);

            for (std::size_t j = 0; j < 5; ++j) {
                ASSERT_TRUE(std::abs(y({j}) - b({j})) < 1.0e-15);
            }
        }

        {
            TensorFact::Array<double> y;
            A.Multiply(true, x, true, y);

            for (std::size_t j = 0; j < 5; ++j) {
                ASSERT_TRUE(std::abs(y({j}) - b({j})) < 1.0e-15);
            }
        }
    }
}

TEST(Array, MatrixMatrixMultiply) {
    {
        TensorFact::Array<float> A;
        A.Resize({2, 3});
        A({0, 0}) = 0.5773502691896258;
        A({1, 0}) = 0.4082482904638631;
        A({0, 1}) = 0.5773502691896258;
        A({1, 1}) = -0.8164965809277261;
        A({0, 2}) = 0.5773502691896258;
        A({1, 2}) = 0.4082482904638631;

        TensorFact::Array<float> B;
        B.Resize({3, 2});
        B({0, 0}) = 0.5773502691896258;
        B({1, 0}) = 0.5773502691896258;
        B({2, 0}) = 0.5773502691896258;
        B({0, 1}) = 0.4082482904638631;
        B({1, 1}) = -0.8164965809277261;
        B({2, 1}) = 0.4082482904638631;

        {
            TensorFact::Array<float> C;
            A.Multiply(false, B, false, C);

            ASSERT_TRUE(std::abs(C({0, 0}) - 1.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({1, 0}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({0, 1}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({1, 1}) - 1.0) < 1.0e-06);
        }

        {
            TensorFact::Array<float> C;
            A.Multiply(true, B, true, C);

            ASSERT_TRUE(std::abs(C({0, 0}) - 0.5) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({1, 0}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({2, 0}) - 0.5) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({0, 1}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({1, 1}) - 1.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({2, 1}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({0, 2}) - 0.5) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({1, 2}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({2, 2}) - 0.5) < 1.0e-06);
        }
    }

    {
        TensorFact::Array<float> A;
        A.Resize({3, 2});
        A({0, 0}) = 0.5773502691896258;
        A({1, 0}) = 0.5773502691896258;
        A({2, 0}) = 0.5773502691896258;
        A({0, 1}) = 0.4082482904638631;
        A({1, 1}) = -0.8164965809277261;
        A({2, 1}) = 0.4082482904638631;

        TensorFact::Array<float> B;
        B.Resize({3, 2});
        B({0, 0}) = 0.5773502691896258;
        B({1, 0}) = 0.5773502691896258;
        B({2, 0}) = 0.5773502691896258;
        B({0, 1}) = 0.4082482904638631;
        B({1, 1}) = -0.8164965809277261;
        B({2, 1}) = 0.4082482904638631;

        {
            TensorFact::Array<float> C;
            A.Multiply(true, B, false, C);

            ASSERT_TRUE(std::abs(C({0, 0}) - 1.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({1, 0}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({0, 1}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({1, 1}) - 1.0) < 1.0e-06);
        }

        {
            TensorFact::Array<float> C;
            A.Multiply(false, B, true, C);

            ASSERT_TRUE(std::abs(C({0, 0}) - 0.5) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({1, 0}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({2, 0}) - 0.5) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({0, 1}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({1, 1}) - 1.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({2, 1}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({0, 2}) - 0.5) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({1, 2}) - 0.0) < 1.0e-06);
            ASSERT_TRUE(std::abs(C({2, 2}) - 0.5) < 1.0e-06);
        }
    }
}

TEST(Array, ReducedRq) {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    TensorFact::Array<double> A;
    A.Resize({3, 4});
    for (std::size_t j = 0; j < 4; ++j) {
        for (std::size_t i = 0; i < 3; ++i) {
            A({i, j}) = distribution(generator);
        }
    }

    TensorFact::Array<double> R;
    TensorFact::Array<double> Q;
    A.ReducedRq(R, Q);

    ASSERT_EQ(R.NDim(), 2);
    ASSERT_EQ(R.Size(0), 3);
    ASSERT_EQ(R.Size(1), 3);

    ASSERT_EQ(Q.NDim(), 2);
    ASSERT_EQ(Q.Size(0), 3);
    ASSERT_EQ(Q.Size(1), 4);

    TensorFact::Array<double> B;
    R.Multiply(false, Q, false, B);

    ASSERT_TRUE((A - B).FrobeniusNorm() < 1.0e-15);

    for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t i = j + 1;i < 3; ++i) {
            ASSERT_TRUE(std::abs(R({i, j})) < 1.0e-15);
        }
    }

    for (std::size_t j = 0; j < 3; ++j) {
        for (std::size_t i = 0; i < j; ++i) {
            double dot_product = 0.0;
            for (std::size_t k = 0; k < 4; ++k) {
                dot_product += Q({i, k}) * Q({j, k});
            }

            ASSERT_TRUE(std::abs(dot_product) < 1.0e-15);
        }

        double norm_squared = 0.0;
        for (std::size_t k = 0; k < 4; ++k) {
            norm_squared += std::pow(Q({j, k}), 2.0);
        }

        ASSERT_TRUE(std::abs(norm_squared - 1.0) < 1.0e-15);
    }
}

TEST(Array, TruncatedSvd) {
    TensorFact::Array<double> A;
    A.Resize({4, 3});
    A({0, 0}) = 0.8730758679812514;
    A({1, 0}) = 0.8905818857769324;
    A({2, 0}) = 1.0004504376390704;
    A({3, 0}) = 1.2121943608516716;
    A({0, 1}) = 1.039764271321522;
    A({1, 1}) = 0.9910014553992643;
    A({2, 1}) = 0.7262655727821424;
    A({3, 1}) = 0.9997665065853555;
    A({0, 2}) = 0.5079667994149603;
    A({1, 2}) = 0.6486914410114907;
    A({2, 2}) = 1.1665137875687934;
    A({3, 2}) = 1.023738282971006;

    TensorFact::Array<double> U_true;
    U_true.Resize({4, 3});
    U_true({0, 0}) = -0.4374561669866752;
    U_true({1, 0}) = -0.4552544328197976;
    U_true({2, 0}) = -0.514262977748206;
    U_true({3, 0}) = -0.5804387074280961;
    U_true({0, 1}) = -0.6166991646254513;
    U_true({1, 1}) = -0.361342760559755;
    U_true({2, 1}) = 0.6849330008504085;
    U_true({3, 1}) = 0.14135180963603095;
    U_true({0, 2}) = -0.10536900266635602;
    U_true({1, 2}) = -0.3861054867694944;
    U_true({2, 2}) = -0.4619084065205092;
    U_true({3, 2}) = 0.7914926091564425;

    TensorFact::Array<double> s_true;
    s_true.Resize({3});
    s_true({0}) = 3.2265338169944235;
    s_true({1}) = 0.5355825867628933;
    s_true({2}) = 0.07847516615102999;

    TensorFact::Array<double> V_true;
    V_true.Resize({3, 3});
    V_true({0, 0}) = -0.6215564555795905;
    V_true({1, 0}) = -0.5764091990342418;
    V_true({2, 0}) = -0.5304903465625369;
    V_true({0, 1}) = -0.0068021041920685536;
    V_true({1, 1}) = -0.673193835938418;
    V_true({2, 1}) = 0.7394347778087524;
    V_true({0, 2}) = 0.7833398393455403;
    V_true({1, 2}) = -0.46320891023729516;
    V_true({2, 2}) = -0.4145071791548137;

    {
        TensorFact::Array<double> U, s, Vt;
        A.TruncatedSvd(U, s, Vt, 0.0, false);

        ASSERT_TRUE(s.NumberOfElements() == 3);

        for (std::size_t r = 0; r < 3; ++r) {
            if (U({0, r}) * U_true({0, r}) < 0.0) {
                for (std::size_t i = 0; i < 4; ++i) {
                    U({i, r}) = -U({i, r});
                }
                for (std::size_t j = 0; j < 3; ++j) {
                    Vt({r, j}) = -Vt({r, j});
                }
            }
        }

        for (std::size_t r = 0; r < 3; ++r) {
            for (std::size_t i = 0; i < 4; ++i) {
                EXPECT_TRUE(std::abs(U({i, r}) - U_true({i, r})) < 1.0e-15);
            }
        }

        for (std::size_t r = 0; r < 3; ++r) {
            ASSERT_TRUE(std::abs(s({r}) - s_true({r})) < 1.0e-15);
        }

        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t r = 0; r < 3; ++r) {
                ASSERT_TRUE(std::abs(Vt({r, j}) - V_true({j, r})) < 1.0e-15);
            }
        }
    }

    {
        TensorFact::Array<double> U, s, Vt;
        A.TruncatedSvd(U, s, Vt, 1.0e-01, false);

        const std::size_t rank = s.NumberOfElements();

        ASSERT_TRUE(rank == 2);

        for (std::size_t r = 0; r < rank; ++r) {
            if (U({0, r}) * U_true({0, r}) < 0.0) {
                for (std::size_t i = 0; i < 4; ++i) {
                    U({i, r}) = -U({i, r});
                }
                for (std::size_t j = 0; j < 3; ++j) {
                    Vt({r, j}) = -Vt({r, j});
                }
            }
        }

        for (std::size_t r = 0; r < rank; ++r) {
            for (std::size_t i = 0; i < 4; ++i) {
                EXPECT_TRUE(std::abs(U({i, r}) - U_true({i, r})) < 1.0e-15);
            }
        }

        for (std::size_t r = 0; r < rank; ++r) {
            ASSERT_TRUE(std::abs(s({r}) - s_true({r})) < 1.0e-15);
        }

        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t r = 0; r < rank; ++r) {
                ASSERT_TRUE(std::abs(Vt({r, j}) - V_true({j, r})) < 1.0e-15);
            }
        }
    }

    {
        TensorFact::Array<double> U, s, Vt;
        A.TruncatedSvd(U, s, Vt, 2.0e-01, true);

        const std::size_t rank = s.NumberOfElements();

        ASSERT_TRUE(rank == 1);

        for (std::size_t r = 0; r < rank; ++r) {
            if (U({0, r}) * U_true({0, r}) < 0.0) {
                for (std::size_t i = 0; i < 4; ++i) {
                    U({i, r}) = -U({i, r});
                }
                for (std::size_t j = 0; j < 3; ++j) {
                    Vt({r, j}) = -Vt({r, j});
                }
            }
        }

        for (std::size_t r = 0; r < rank; ++r) {
            for (std::size_t i = 0; i < 4; ++i) {
                EXPECT_TRUE(std::abs(U({i, r}) - U_true({i, r})) < 1.0e-15);
            }
        }

        for (std::size_t r = 0; r < rank; ++r) {
            ASSERT_TRUE(std::abs(s({r}) - s_true({r})) < 1.0e-15);
        }

        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t r = 0; r < rank; ++r) {
                ASSERT_TRUE(std::abs(Vt({r, j}) - V_true({j, r})) < 1.0e-15);
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
