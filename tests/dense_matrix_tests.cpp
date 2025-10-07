/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Yaroslav Riabtsev <yaroslav.riabtsev@rwth-aachen.de>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "dense_matrix.hpp"
#include <gtest/gtest.h>

using dm::dense_matrix;
using dm::mul_algo;

TEST(DenseMatrix, BasicCtorAndAccess) {
    dense_matrix<double> m(2, 3, { 1, 2, 3, 4, 5, 6 });
    EXPECT_EQ(m.rows(), 2u);
    EXPECT_EQ(m.cols(), 3u);
    EXPECT_DOUBLE_EQ(m.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.at(1, 2), 6.0);
    m(0, 1) = 42.0;
    EXPECT_DOUBLE_EQ(m.at(0, 1), 42.0);
}

TEST(DenseMatrix, MatMul_2x3_3x2_Int_AllAlgos) {
    dense_matrix<int> a(2, 3, { 1, 2, 3, 4, 5, 6 });
    dense_matrix<int> b(3, 2, { 7, 8, 9, 10, 11, 12 });
    dense_matrix<int> e(2, 2, { 58, 64, 139, 154 });

    auto c_native = dense_matrix<int>::multiply(a, b, mul_algo::native);
    auto c_transp = dense_matrix<int>::multiply(a, b, mul_algo::transpose);
    auto c_ijp = dense_matrix<int>::multiply(a, b, mul_algo::block_ijp, 2);
    auto c_ipj = dense_matrix<int>::multiply(a, b, mul_algo::block_ipj, 2);

    EXPECT_TRUE(c_native == e);
    EXPECT_TRUE(c_transp == e);
    EXPECT_TRUE(c_ijp == e);
    EXPECT_TRUE(c_ipj == e);
}

TEST(DenseMatrix, MatMul_2x3_3x2_Double_AllAlgos) {
    dense_matrix<double> a(2, 3, { 1, 2, 3, 4, 5, 6 });
    dense_matrix<double> b(3, 2, { 7, 8, 9, 10, 11, 12 });
    dense_matrix<double> e(2, 2, { 58, 64, 139, 154 });

    auto c_native = dense_matrix<double>::multiply(a, b, mul_algo::native);
    auto c_transp = dense_matrix<double>::multiply(a, b, mul_algo::transpose);
    auto c_ijp = dense_matrix<double>::multiply(a, b, mul_algo::block_ijp, 2);
    auto c_ipj = dense_matrix<double>::multiply(a, b, mul_algo::block_ipj, 2);

    for (size_t i = 0; i < e.size(); ++i) {
        EXPECT_DOUBLE_EQ(c_native.data()[i], e.data()[i]);
        EXPECT_DOUBLE_EQ(c_transp.data()[i], e.data()[i]);
        EXPECT_DOUBLE_EQ(c_ijp.data()[i], e.data()[i]);
        EXPECT_DOUBLE_EQ(c_ipj.data()[i], e.data()[i]);
    }
}

TEST(DenseMatrix, IdentityRight) {
    dense_matrix<int> a(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    dense_matrix<int> b(3, 3);
    for (size_t d = 0; d < 3; ++d) {
        b(d, d) = 1;
    }

    auto c_native = dense_matrix<int>::multiply(a, b, mul_algo::native);
    auto c_ijp = dense_matrix<int>::multiply(a, b, mul_algo::block_ijp, 0);
    EXPECT_TRUE(c_native == a);
    EXPECT_TRUE(c_ijp == a);
}

TEST(DenseMatrix, ZeroDims) {
    dense_matrix<double> a(0, 5);
    dense_matrix<double> b(5, 0);
    auto c = dense_matrix<double>::multiply(a, b, mul_algo::native);
    EXPECT_EQ(c.rows(), 0u);
    EXPECT_EQ(c.cols(), 0u);
}

TEST(DenseMatrix, MismatchThrows) {
    dense_matrix<int> a(2, 3, { 1, 2, 3, 4, 5, 6 });
    dense_matrix<int> b(4, 2, { 1, 2, 3, 4, 5, 6, 7, 8 });
    EXPECT_THROW(
        (void)dense_matrix<int>::multiply(a, b, mul_algo::native),
        std::invalid_argument
    );
}

TEST(DenseMatrix, AtThrows) {
    dense_matrix<int> m(1, 1, { 7 });
    EXPECT_THROW(m.at(1, 0), std::out_of_range);
    EXPECT_THROW(m.at(0, 1), std::out_of_range);
}

TEST(DenseMatrix, SafeCountOverflow) {
    const size_t m = std::numeric_limits<size_t>::max();
    EXPECT_THROW((dense_matrix<int>(m, 2)), std::overflow_error);
}

TEST(DenseMatrix_Nested, Multiply_1x2_by_2x1) {
    dense_matrix<double> a00(2, 3, { 1, 2, 3, 4, 5, 6 });
    dense_matrix<double> a01(2, 3, { -1, 0, 2, 1, -2, 3 });
    dense_matrix<double> b00(3, 4, { 1, 0, 2, -1, 0, 1, -1, 2, 2, -1, 0, 1 });
    dense_matrix<double> b10(
        3, 4, { 0.5, 1, -1.5, 0, 1, 0, 1, -1, -2, 1, 0, 2 }
    );

    dense_matrix<dense_matrix<double>> a(1, 2, { a00, a01 });
    dense_matrix<dense_matrix<double>> b(2, 1, { b00, b10 });

    auto c_native
        = dense_matrix<dense_matrix<double>>::multiply(a, b, mul_algo::native);
    auto c_transp = dense_matrix<dense_matrix<double>>::multiply(
        a, b, mul_algo::transpose, 8
    );
    auto c_ijp = dense_matrix<dense_matrix<double>>::multiply(
        a, b, mul_algo::block_ijp, 8
    );
    auto c_ipj = dense_matrix<dense_matrix<double>>::multiply(
        a, b, mul_algo::block_ipj, 8
    );

    dense_matrix<double> expected = (a00 * b00) + (a01 * b10);

    ASSERT_EQ(c_native.rows(), 1u);
    ASSERT_EQ(c_native.cols(), 1u);

    EXPECT_TRUE(c_native.at(0, 0) == expected);
    EXPECT_TRUE(c_transp.at(0, 0) == expected);
    EXPECT_TRUE(c_ijp.at(0, 0) == expected);
    EXPECT_TRUE(c_ipj.at(0, 0) == expected);
}
