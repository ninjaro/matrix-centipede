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
#include "dense_matrix_api.h"
#include <gtest/gtest.h>
#include <numeric>

TEST(DenseMatrixApiTest, CreateMatrixTest) {
    auto* obj = dm_new(2, 3);
    ASSERT_NE(obj, nullptr);

    EXPECT_EQ(dm_rows(obj), 2u);
    EXPECT_EQ(dm_cols(obj), 3u);
    EXPECT_EQ(dm_size(obj), 6u);

    dm_delete(obj);
}

TEST(DenseMatrixApiTest, MultiplyTest) {
    auto* lhs = dm_new(2, 3);
    auto* rhs = dm_new(3, 2);
    ASSERT_NE(lhs, nullptr);
    ASSERT_NE(rhs, nullptr);

    std::array<double, 12> data {};
    std::iota(data.begin(), data.end(), 1.0);

    ASSERT_EQ(dm_write(lhs, data.data(), 6), ok);
    ASSERT_EQ(dm_write(rhs, data.data() + 6, 6), ok);

    dm_storage* out = nullptr;
    ASSERT_EQ(dm_mul(lhs, rhs, &out), ok);
    ASSERT_NE(out, nullptr);

    std::array<double, 4> actual {};
    ASSERT_EQ(dm_read(out, actual.data(), actual.size()), ok);

    const std::array<double, 4> expected { 58.0, 64.0, 139.0, 154.0 };
    EXPECT_EQ(actual, expected);

    dm_delete(out);
    dm_delete(lhs);
    dm_delete(rhs);
}