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

TEST(DenseMatrix, EmptyConstruct) {
    dense_matrix m;
    EXPECT_EQ(m.rows(), 0u);
    EXPECT_EQ(m.cols(), 0u);
    EXPECT_EQ(m.size(), 0u);
}

TEST(DenseMatrix, InitListConstructAndAccess) {
    dense_matrix<double> m(2, 3, { 1, 2, 3, 4, 5, 6 });
    EXPECT_EQ(m.rows(), 2u);
    EXPECT_EQ(m.cols(), 3u);
    EXPECT_DOUBLE_EQ(m.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.at(1, 2), 6.0);
    m(0, 1) = 42.0;
    EXPECT_DOUBLE_EQ(m.at(0, 1), 42.0);
}

TEST(DenseMatrix, PointerAndSpanCtor) {
    const double buf[6] = { 1, 2, 3, 4, 5, 6 };
    dense_matrix a(2, 3, buf);
    EXPECT_DOUBLE_EQ(a.at(1, 1), 5.0);
}

TEST(DenseMatrix, AtThrows) {
    dense_matrix<int> m(1, 1, { 7 });
    EXPECT_THROW(m.at(1, 0), std::out_of_range);
    EXPECT_THROW(m.at(0, 1), std::out_of_range);
}
