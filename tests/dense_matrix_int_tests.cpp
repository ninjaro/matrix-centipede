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

TEST(DenseIntMatrix, CreateMatrix) {
    dm_int m1(2, 3);

    EXPECT_EQ(m1.rows(), 2u);
    EXPECT_EQ(m1.cols(), 3u);
    EXPECT_EQ(m1.size(), 6u);

    dm_int m2(m1);
    EXPECT_TRUE(m1 == m2);
    const dm_int m3(std::move(m1));
    EXPECT_TRUE(m2 == m3);
    dm_int m4 = m3;
    EXPECT_TRUE(m2 == m4);
    dm_int m5 = std::move(m2);
    EXPECT_TRUE(m4 == m5);
}

TEST(DenseIntMatrix, MatrixAccess) {
    int tmp = 42;
    dm_int m1(2, 3);
    m1.at(0, 0) = tmp;
    tmp = m1.at(1, 1);
    m1(1, 0) = tmp;

    EXPECT_THROW(m1.at(2, 0), std::out_of_range);
    EXPECT_THROW(m1.at(0, 3), std::out_of_range);
    EXPECT_THROW(m1.at(2, 3), std::out_of_range);
}

TEST(DenseIntMatrix, CreateZeroMatrix) {
    dm_int m1;
    dm_int m2(0, 0);
    dm_int m3(0, 0, {});
    dm_int m4(0, 0, std::span<const int> {});

    EXPECT_TRUE(m1 == m2);
    EXPECT_TRUE(m2 == m3);
}

TEST(DenseIntMatrix, CreateWithNull) {
    EXPECT_NO_THROW((dm_int(0, 0, nullptr)));
    EXPECT_THROW((dm_int(2, 3, nullptr)), std::invalid_argument);
}

TEST(DenseIntMatrix, CreateSizeMismatch) {
    EXPECT_THROW((dm_int(2, 2, { 1, 2, 3, 4, 5, 6 })), std::invalid_argument);
    EXPECT_THROW(
        (dm_int(2, 2, std::span<const int>({ 1, 2, 3, 4, 5, 6 }))),
        std::invalid_argument
    );

    // std::array<int, 6> data {};
    // std::iota(data.begin(), data.end(), 1.0);
    // EXPECT_DEATH((dm_int(3, 3, data.data())), "");
}

TEST(DenseIntMatrix, BasicCreateAndAccess) {
    dm_int m1(2, 3, { 1, 2, 3, 4, 5, 6 });
    dm_int m2(2, 3, std::span<const int>({ 1, 2, 3, 4, 5, 6 }));

    std::array<int, 6> data {};
    std::iota(data.begin(), data.end(), 1);
    dm_int m3(2, 3, data.data());

    EXPECT_TRUE(m1 == m2);
    EXPECT_TRUE(m2 == m3);

    std::array<int, 6> m1_data {};
    std::copy_n(m1.data(), m1_data.size(), m1_data.begin());
    EXPECT_EQ(m1_data, data);

    const dm_int& cm2 = m2;
    EXPECT_EQ(cm2.data(), m2.data());
}

TEST(DenseIntMatrix, MatMul) {
    dm_int a(2, 3, { 1, 2, 3, 4, 5, 6 });
    dm_int b(3, 2, { 7, 8, 9, 10, 11, 12 });
    dm_int e(2, 2, { 58, 64, 139, 154 });

    auto c_native = dm_int::multiply(a, b, mul_algo::native);
    auto c_transp = dm_int::multiply(a, b, mul_algo::transpose);
    auto c_ijp = dm_int::multiply(a, b, mul_algo::block_ijp);
    auto c_ipj = dm_int::multiply(a, b, mul_algo::block_ipj);

    EXPECT_TRUE(c_native == e);
    EXPECT_TRUE(c_transp == e);
    EXPECT_TRUE(c_ijp == e);
    EXPECT_TRUE(c_ipj == e);
}

TEST(DenseIntMatrix, MatSmallMul) {
    for (size_t i = 0; i < 3; ++i) {
        dm_int a(2, i);
        dm_int b(i, 2);

        EXPECT_NO_THROW((void)(dm_int::multiply(a, b, mul_algo::native, i)));
        EXPECT_NO_THROW((void)(dm_int::multiply(b, a, mul_algo::native, i)));
        EXPECT_NO_THROW((void)(dm_int::multiply(a, b, mul_algo::transpose, i)));
        EXPECT_NO_THROW((void)(dm_int::multiply(b, a, mul_algo::transpose, i)));
        EXPECT_NO_THROW((void)(dm_int::multiply(a, b, mul_algo::block_ijp, i)));
        EXPECT_NO_THROW((void)(dm_int::multiply(b, a, mul_algo::block_ijp, i)));
        EXPECT_NO_THROW((void)(dm_int::multiply(a, b, mul_algo::block_ipj, i)));
        EXPECT_NO_THROW((void)(dm_int::multiply(b, a, mul_algo::block_ipj, i)));
    }
}

TEST(DenseIntMatrix, MismatchThrows) {
    dm_int a(2, 3);
    dm_int b(4, 2);
    EXPECT_THROW((void)(a * b), std::invalid_argument);
}

TEST(DenseIntMatrix, MatrixComparation) {
    dm_int m1(2, 3);
    dm_int m2(4, 2);
    dm_int m3(4, 3);
    dm_int m4(2, 3);
    m4(1, 2) = 1;
    EXPECT_NE(m1, m2);
    EXPECT_NE(m2, m3);
    EXPECT_NE(m1, m3);
    EXPECT_NE(m1, m4);
}

TEST(DenseIntMatrix, SafeCountOverflow) {
    const size_t m = std::numeric_limits<size_t>::max();
    EXPECT_THROW((dm_int(m, 2)), std::overflow_error);
}
