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
#include <numeric>

#include "dense_matrix.hpp"
#include <gtest/gtest.h>

using dm::dense_matrix;
using dm::mul_algo;

typedef dense_matrix<int> dm_int;

TEST(DenseMatrix, TensorMultiply) {
    dm_int a00(2, 3, { 1, 1, 2, 3, 5, 8 });
    dm_int a01(2, 3, { 1, 2, 3, 4, 5, 6 });
    dm_int b00(3, 4, { 1, 3, 5, 8, 10, 14, 16, 20, 23, 27, 29, 35 });
    dm_int b10(3, 4, { 1, 4, 8, 8, 6, 2, 8, 7, 7, 2, 9, 7 });

    dense_matrix<dm_int> a(1, 2, { a00, a01 });
    dense_matrix<dm_int> b(2, 1, { b00, b10 });

    auto c_native = dense_matrix<dm_int>::multiply(a, b, mul_algo::native);
    auto c_transp
        = dense_matrix<dm_int>::multiply(a, b, mul_algo::transpose, 8);
    auto c_ijp = dense_matrix<dm_int>::multiply(a, b, mul_algo::block_ijp, 8);
    auto c_ipj = dense_matrix<dm_int>::multiply(a, b, mul_algo::block_ipj, 8);

    dm_int expected = (a00 * b00) + (a01 * b10);

    ASSERT_EQ(c_native.rows(), 1u);
    ASSERT_EQ(c_native.cols(), 1u);

    EXPECT_TRUE(c_native.at(0, 0) == expected);
    EXPECT_TRUE(c_transp.at(0, 0) == expected);
    EXPECT_TRUE(c_ijp.at(0, 0) == expected);
    EXPECT_TRUE(c_ipj.at(0, 0) == expected);
}
