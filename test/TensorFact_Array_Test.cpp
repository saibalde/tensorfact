#include "TensorFact_Array.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

template <typename Scalar>
TensorFact::Array<Scalar> CreateRandomArray(
    const std::vector<std::size_t> &size) {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<Scalar> distribution(-1.0, 1.0);

    TensorFact::Array<Scalar> A;
    A.Resize(size);

    const std::size_t num_element = A.NumberOfElements();

    for (std::size_t n = 0; n < num_element; ++n) {
        A(n) = distribution(generator);
    }

    return A;
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

template <typename Scalar>
void MatrixVectorMultiplyTest(std::size_t m, std::size_t n, bool conjugate) {
    const TensorFact::Array<Scalar> A = CreateRandomArray<Scalar>({m, n});
    TensorFact::Array<Scalar> x;
    TensorFact::Array<Scalar> b_expected;

    if (!conjugate) {
        x = CreateRandomArray<Scalar>({n});

        b_expected.Resize({m});
        for (std::size_t i = 0; i < m; ++i) {
            b_expected({i}) = static_cast<Scalar>(0);
        }
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                b_expected({i}) += A({i, j}) * x({j});
            }
        }
    } else {
        x = CreateRandomArray<Scalar>({m});

        b_expected.Resize({n});
        for (std::size_t j = 0; j < n; ++j) {
            b_expected({j}) = static_cast<Scalar>(0);
        }
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                b_expected({j}) += A({i, j}) * x({i});
            }
        }
    }

    TensorFact::Array<Scalar> b1;
    A.Multiply(conjugate, x, false, b1);
    ASSERT_TRUE((b_expected - b1).FrobeniusNorm() <
                100 * std::numeric_limits<Scalar>::epsilon());

    TensorFact::Array<Scalar> b2;
    A.Multiply(conjugate, x, true, b2);
    ASSERT_TRUE((b_expected - b2).FrobeniusNorm() <
                100 * std::numeric_limits<Scalar>::epsilon());
}

TEST(Array, MatrixVectorMultiply) {
    MatrixVectorMultiplyTest<double>(64, 4, true);
    MatrixVectorMultiplyTest<double>(64, 4, false);

    MatrixVectorMultiplyTest<float>(64, 4, true);
    MatrixVectorMultiplyTest<float>(64, 4, false);

    MatrixVectorMultiplyTest<double>(4, 64, true);
    MatrixVectorMultiplyTest<double>(4, 64, false);

    MatrixVectorMultiplyTest<float>(4, 64, true);
    MatrixVectorMultiplyTest<float>(4, 64, false);
}

template <typename Scalar>
void MatrixMatrixMultiplyTest(std::size_t p, std::size_t q, std::size_t r,
                              bool conjugate_A, bool conjugate_B) {
    TensorFact::Array<Scalar> A = CreateRandomArray<Scalar>({p, q});
    TensorFact::Array<Scalar> B;
    TensorFact::Array<Scalar> C_expected;

    if (!conjugate_A && !conjugate_B) {
        B = CreateRandomArray<Scalar>({q, r});

        C_expected.Resize({p, r});
        for (std::size_t k = 0; k < r; ++k) {
            for (std::size_t i = 0; i < p; ++i) {
                C_expected({i, k}) = static_cast<Scalar>(0);
            }
        }
        for (std::size_t k = 0; k < r; ++k) {
            for (std::size_t j = 0; j < q; ++j) {
                for (std::size_t i = 0; i < p; ++i) {
                    C_expected({i, k}) += A({i, j}) * B({j, k});
                }
            }
        }
    } else if (!conjugate_A && conjugate_B) {
        B = CreateRandomArray<Scalar>({r, q});

        C_expected.Resize({p, r});
        for (std::size_t k = 0; k < r; ++k) {
            for (std::size_t i = 0; i < p; ++i) {
                C_expected({i, k}) = static_cast<Scalar>(0);
            }
        }
        for (std::size_t k = 0; k < r; ++k) {
            for (std::size_t j = 0; j < q; ++j) {
                for (std::size_t i = 0; i < p; ++i) {
                    C_expected({i, k}) += A({i, j}) * B({k, j});
                }
            }
        }
    } else if (conjugate_A && !conjugate_B) {
        B = CreateRandomArray<Scalar>({p, r});

        C_expected.Resize({q, r});
        for (std::size_t k = 0; k < r; ++k) {
            for (std::size_t j = 0; j < q; ++j) {
                C_expected({j, k}) = static_cast<Scalar>(0);
            }
        }
        for (std::size_t k = 0; k < r; ++k) {
            for (std::size_t j = 0; j < q; ++j) {
                for (std::size_t i = 0; i < p; ++i) {
                    C_expected({j, k}) += A({i, j}) * B({i, k});
                }
            }
        }
    } else {
        B = CreateRandomArray<Scalar>({r, p});

        C_expected.Resize({q, r});
        for (std::size_t k = 0; k < r; ++k) {
            for (std::size_t j = 0; j < q; ++j) {
                C_expected({j, k}) = static_cast<Scalar>(0);
            }
        }
        for (std::size_t k = 0; k < r; ++k) {
            for (std::size_t j = 0; j < q; ++j) {
                for (std::size_t i = 0; i < p; ++i) {
                    C_expected({j, k}) += A({i, j}) * B({k, i});
                }
            }
        }
    }

    TensorFact::Array<Scalar> C;
    A.Multiply(conjugate_A, B, conjugate_B, C);

    ASSERT_TRUE((C_expected - C).FrobeniusNorm() <
                100 * std::numeric_limits<Scalar>::epsilon());
}

TEST(Array, MatrixMatrixMultiply) {
    {
        const std::size_t p = 32;
        const std::size_t q = 16;
        const std::size_t r = 8;

        MatrixMatrixMultiplyTest<float>(p, q, r, false, false);
        MatrixMatrixMultiplyTest<float>(p, q, r, false, true);
        MatrixMatrixMultiplyTest<float>(p, q, r, true, false);
        MatrixMatrixMultiplyTest<float>(p, q, r, true, true);

        MatrixMatrixMultiplyTest<double>(p, q, r, false, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, false, true);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, true);
    }
    {
        const std::size_t p = 32;
        const std::size_t q = 8;
        const std::size_t r = 16;

        MatrixMatrixMultiplyTest<float>(p, q, r, false, false);
        MatrixMatrixMultiplyTest<float>(p, q, r, false, true);
        MatrixMatrixMultiplyTest<float>(p, q, r, true, false);
        MatrixMatrixMultiplyTest<float>(p, q, r, true, true);

        MatrixMatrixMultiplyTest<double>(p, q, r, false, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, false, true);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, true);
    }
    {
        const std::size_t p = 16;
        const std::size_t q = 8;
        const std::size_t r = 32;

        MatrixMatrixMultiplyTest<float>(p, q, r, false, false);
        MatrixMatrixMultiplyTest<float>(p, q, r, false, true);
        MatrixMatrixMultiplyTest<float>(p, q, r, true, false);
        MatrixMatrixMultiplyTest<float>(p, q, r, true, true);

        MatrixMatrixMultiplyTest<double>(p, q, r, false, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, false, true);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, true);
    }
    {
        const std::size_t p = 16;
        const std::size_t q = 32;
        const std::size_t r = 8;

        MatrixMatrixMultiplyTest<float>(p, q, r, false, false);
        MatrixMatrixMultiplyTest<float>(p, q, r, false, true);
        MatrixMatrixMultiplyTest<float>(p, q, r, true, false);
        MatrixMatrixMultiplyTest<float>(p, q, r, true, true);

        MatrixMatrixMultiplyTest<double>(p, q, r, false, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, false, true);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, true);
    }
    {
        const std::size_t p = 8;
        const std::size_t q = 32;
        const std::size_t r = 16;

        MatrixMatrixMultiplyTest<double>(p, q, r, false, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, false, true);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, true);
    }
    {
        const std::size_t p = 8;
        const std::size_t q = 16;
        const std::size_t r = 32;

        MatrixMatrixMultiplyTest<double>(p, q, r, false, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, false, true);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, false);
        MatrixMatrixMultiplyTest<double>(p, q, r, true, true);
    }
}

template <typename Scalar>
void ReducedRqTest(std::size_t m, std::size_t n) {
    const Scalar epsilon = 100 * std::numeric_limits<Scalar>::epsilon();
    const std::size_t k = std::min(m, n);

    TensorFact::Array<Scalar> A = CreateRandomArray<Scalar>({m, n});

    TensorFact::Array<Scalar> R;
    TensorFact::Array<Scalar> Q;
    A.ReducedRq(R, Q);

    ASSERT_EQ(R.NDim(), 2);
    ASSERT_EQ(R.Size(0), m);
    ASSERT_EQ(R.Size(1), k);

    ASSERT_EQ(Q.NDim(), 2);
    ASSERT_EQ(Q.Size(0), k);
    ASSERT_EQ(Q.Size(1), n);

    for (std::size_t j = 0; j < k; ++j) {
        for (std::size_t i = j + m - k + 1; i < m; ++i) {
            ASSERT_TRUE(std::abs(R({i, j})) < epsilon);
        }
    }

    for (std::size_t i1 = 0; i1 < k; ++i1) {
        for (std::size_t i2 = 0; i2 < i1; ++i2) {
            Scalar dot_product = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                dot_product += Q({i1, j}) * Q({i2, j});
            }

            ASSERT_TRUE(std::abs(dot_product) < epsilon);
        }

        Scalar norm_squared = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            norm_squared += std::pow(Q({i1, j}), 2.0);
        }

        ASSERT_TRUE(std::abs(norm_squared - 1.0) < epsilon);
    }

    TensorFact::Array<Scalar> B;
    R.Multiply(false, Q, false, B);

    ASSERT_TRUE((A - B).FrobeniusNorm() < epsilon);
}

TEST(Array, ReducedRq) {
    ReducedRqTest<float>(4, 64);
    ReducedRqTest<double>(4, 64);

    ReducedRqTest<float>(64, 4);
    ReducedRqTest<double>(64, 4);
}

template <typename Scalar>
void TruncatedSvdTest(std::size_t m, std::size_t n, std::size_t r) {
    const Scalar epsilon = 100 * std::numeric_limits<Scalar>::epsilon();

    // Construct appropriate matrix
    TensorFact::Array<Scalar> A;
    std::size_t k;
    Scalar abs_acc;
    Scalar rel_acc;

    {
        // Create Ut factor
        TensorFact::Array<Scalar> Ut;
        {
            TensorFact::Array<Scalar> temp1 = CreateRandomArray<Scalar>({r, m});
            TensorFact::Array<Scalar> temp2;

            temp1.ReducedRq(temp2, Ut);
        }

        // Create s factor; s(i) ~ Uniform([2 * (r - 1 - i), 2 * (r - i)])
        TensorFact::Array<Scalar> s = CreateRandomArray<Scalar>({r});
        for (std::size_t i = 0; i < r; ++i) {
            s({i}) += 2 * (r - 1 - i) + 1;
        }

        // Create Vt factor
        TensorFact::Array<Scalar> Vt;
        {
            TensorFact::Array<Scalar> temp1 = CreateRandomArray<Scalar>({r, n});
            TensorFact::Array<Scalar> temp2;

            temp1.ReducedRq(temp2, Vt);
        }

        // Compute A = Ut.conjugate() * diagmat(s) * Vt
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < r; ++i) {
                Vt({i, j}) *= s({i});
            }
        }

        Ut.Multiply(true, Vt, false, A);

        // Compute absolute and relative errors given specified maximum rank
        TensorFact::Array<Scalar> absolute_error;
        absolute_error.Resize({r});

        absolute_error({r - 1}) = static_cast<Scalar>(0);
        for (std::size_t i = r - 1; i > 0; --i) {
            absolute_error({i - 1}) = absolute_error({i}) + std::pow(s({i}), 2);
        }

        Scalar frobenius_norm = absolute_error({0}) + std::pow(s({0}), 2);

        for (std::size_t i = 0; i < r; ++i) {
            absolute_error({i}) = std::sqrt(absolute_error({i}));
        }

        frobenius_norm = std::sqrt(frobenius_norm);

        TensorFact::Array<Scalar> relative_error;
        relative_error.Resize({r});

        for (std::size_t i = 0; i < r; ++i) {
            relative_error({i}) = absolute_error({i}) / frobenius_norm;
        }

        // Determine 75% Frobenius norm cutoff rank
        k = 1;
        while (k <= r) {
            if (relative_error({k - 1}) <= 0.25) {
                break;
            }
            ++k;
        }

        // Compute accuracy levels based on cutoff rank
        abs_acc = (absolute_error({k - 2}) + absolute_error({k - 1})) / 2;
        rel_acc = (relative_error({k - 2}) + relative_error({k - 1})) / 2;
    }

    // Test thin SVD
    {
        const std::size_t p = std::min(m, n);

        TensorFact::Array<Scalar> U;
        TensorFact::Array<Scalar> s;
        TensorFact::Array<Scalar> Vt;
        A.TruncatedSvd(U, s, Vt, static_cast<Scalar>(0), false);

        ASSERT_EQ(U.NDim(), 2);
        ASSERT_EQ(U.Size(0), m);
        ASSERT_EQ(U.Size(1), p);

        ASSERT_EQ(s.NDim(), 1);
        ASSERT_EQ(s.NumberOfElements(), p);

        ASSERT_EQ(Vt.NDim(), 2);
        ASSERT_EQ(Vt.Size(0), p);
        ASSERT_EQ(Vt.Size(1), n);

        for (std::size_t j1 = 0; j1 < p; ++j1) {
            for (std::size_t j2 = 0; j2 < j1; ++j2) {
                Scalar dot_product = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    dot_product += U({i, j1}) * U({i, j2});
                }

                ASSERT_TRUE(std::abs(dot_product) < epsilon);
            }

            Scalar norm_squared = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                norm_squared += std::pow(U({i, j1}), 2);
            }

            ASSERT_TRUE(std::abs(norm_squared - static_cast<Scalar>(1)) <
                        epsilon);
        }

        for (std::size_t i = 0; i < p; ++i) {
            if (i < p - 1) {
                ASSERT_GE(s({i}), s({i + 1}));
            } else {
                ASSERT_GT(s({i}), static_cast<Scalar>(0));
            }
        }

        for (std::size_t i1 = 0; i1 < p; ++i1) {
            for (std::size_t i2 = 0; i2 < i1; ++i2) {
                Scalar dot_product = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    dot_product += Vt({i1, j}) * Vt({i2, j});
                }

                ASSERT_TRUE(std::abs(dot_product) < epsilon);
            }

            Scalar norm_squared = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                norm_squared += std::pow(Vt({i1, j}), 2);
            }

            ASSERT_TRUE(std::abs(norm_squared - static_cast<Scalar>(1)) <
                        epsilon);
        }

        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < p; ++i) {
                Vt({i, j}) *= s({i});
            }
        }

        TensorFact::Array<Scalar> B;
        U.Multiply(false, Vt, false, B);

        ASSERT_LE((A - B).FrobeniusNorm(), 10 * epsilon);
    }

    // Test truncated SVD with absolute error
    {
        TensorFact::Array<Scalar> U;
        TensorFact::Array<Scalar> s;
        TensorFact::Array<Scalar> Vt;
        A.TruncatedSvd(U, s, Vt, abs_acc, false);

        ASSERT_EQ(U.NDim(), 2);
        ASSERT_EQ(U.Size(0), m);
        ASSERT_EQ(U.Size(1), k);

        ASSERT_EQ(s.NDim(), 1);
        ASSERT_EQ(s.NumberOfElements(), k);

        ASSERT_EQ(Vt.NDim(), 2);
        ASSERT_EQ(Vt.Size(0), k);
        ASSERT_EQ(Vt.Size(1), n);

        for (std::size_t j1 = 0; j1 < k; ++j1) {
            for (std::size_t j2 = 0; j2 < j1; ++j2) {
                Scalar dot_product = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    dot_product += U({i, j1}) * U({i, j2});
                }

                ASSERT_TRUE(std::abs(dot_product) < epsilon);
            }

            Scalar norm_squared = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                norm_squared += std::pow(U({i, j1}), 2);
            }

            ASSERT_TRUE(std::abs(norm_squared - static_cast<Scalar>(1)) <
                        epsilon);
        }

        for (std::size_t i = 0; i < k; ++i) {
            if (i < k - 1) {
                ASSERT_GE(s({i}), s({i + 1}));
            } else {
                ASSERT_GT(s({i}), static_cast<Scalar>(0));
            }
        }

        for (std::size_t i1 = 0; i1 < k; ++i1) {
            for (std::size_t i2 = 0; i2 < i1; ++i2) {
                Scalar dot_product = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    dot_product += Vt({i1, j}) * Vt({i2, j});
                }

                ASSERT_TRUE(std::abs(dot_product) < epsilon);
            }

            Scalar norm_squared = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                norm_squared += std::pow(Vt({i1, j}), 2);
            }

            ASSERT_TRUE(std::abs(norm_squared - static_cast<Scalar>(1)) <
                        epsilon);
        }

        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < k; ++i) {
                Vt({i, j}) *= s({i});
            }
        }

        TensorFact::Array<Scalar> B;
        U.Multiply(false, Vt, false, B);

        ASSERT_LE((A - B).FrobeniusNorm(), abs_acc);
    }

    // Test thin SVD
    {
        TensorFact::Array<Scalar> U;
        TensorFact::Array<Scalar> s;
        TensorFact::Array<Scalar> Vt;
        A.TruncatedSvd(U, s, Vt, rel_acc, true);

        ASSERT_EQ(U.NDim(), 2);
        ASSERT_EQ(U.Size(0), m);
        ASSERT_EQ(U.Size(1), k);

        ASSERT_EQ(s.NDim(), 1);
        ASSERT_EQ(s.NumberOfElements(), k);

        ASSERT_EQ(Vt.NDim(), 2);
        ASSERT_EQ(Vt.Size(0), k);
        ASSERT_EQ(Vt.Size(1), n);

        for (std::size_t j1 = 0; j1 < k; ++j1) {
            for (std::size_t j2 = 0; j2 < j1; ++j2) {
                Scalar dot_product = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    dot_product += U({i, j1}) * U({i, j2});
                }

                ASSERT_TRUE(std::abs(dot_product) < epsilon);
            }

            Scalar norm_squared = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                norm_squared += std::pow(U({i, j1}), 2);
            }

            ASSERT_TRUE(std::abs(norm_squared - static_cast<Scalar>(1)) <
                        epsilon);
        }

        for (std::size_t i = 0; i < k; ++i) {
            if (i < k - 1) {
                ASSERT_GE(s({i}), s({i + 1}));
            } else {
                ASSERT_GT(s({i}), static_cast<Scalar>(0));
            }
        }

        for (std::size_t i1 = 0; i1 < k; ++i1) {
            for (std::size_t i2 = 0; i2 < i1; ++i2) {
                Scalar dot_product = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    dot_product += Vt({i1, j}) * Vt({i2, j});
                }

                ASSERT_TRUE(std::abs(dot_product) < epsilon);
            }

            Scalar norm_squared = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                norm_squared += std::pow(Vt({i1, j}), 2);
            }

            ASSERT_TRUE(std::abs(norm_squared - static_cast<Scalar>(1)) <
                        epsilon);
        }

        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < k; ++i) {
                Vt({i, j}) *= s({i});
            }
        }

        TensorFact::Array<Scalar> B;
        U.Multiply(false, Vt, false, B);

        ASSERT_LE((A - B).FrobeniusNorm(), rel_acc * A.FrobeniusNorm());
    }
}

TEST(Array, TruncatedSvd) {
    TruncatedSvdTest<float>(16, 64, 8);
    TruncatedSvdTest<double>(16, 64, 8);

    TruncatedSvdTest<float>(64, 16, 8);
    TruncatedSvdTest<double>(64, 16, 8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
