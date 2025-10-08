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

#include "dense_matrix_jni.h"
#include <gtest/gtest.h>

dm_ptr to_ptr(jlong obj) {
    return reinterpret_cast<dm_ptr>(static_cast<std::uintptr_t>(obj));
}

jlong to_obj(dm_ptr ptr) {
    return static_cast<jlong>(reinterpret_cast<std::uintptr_t>(ptr));
}

TEST(DenseMatrixJniTest, CreateMatrixTest) {
    jlong obj = Java_dm_DenseMatrixJni_nativeNew(nullptr, nullptr, 2, 3);
    ASSERT_NE(obj, 0);

    EXPECT_EQ(Java_dm_DenseMatrixJni_nativeRows(nullptr, nullptr, obj), 2);
    EXPECT_EQ(Java_dm_DenseMatrixJni_nativeCols(nullptr, nullptr, obj), 3);
    EXPECT_EQ(Java_dm_DenseMatrixJni_nativeSize(nullptr, nullptr, obj), 6);

    Java_dm_DenseMatrixJni_nativeDelete(nullptr, nullptr, obj);
}

TEST(DenseMatrixJniTest, MultiplyTest) {
    jlong lhr = Java_dm_DenseMatrixJni_nativeNew(nullptr, nullptr, 2, 3);
    jlong rhr = Java_dm_DenseMatrixJni_nativeNew(nullptr, nullptr, 3, 2);
    ASSERT_NE(lhr, 0);
    ASSERT_NE(rhr, 0);

    std::array<double, 12> data {};
    std::iota(data.begin(), data.end(), 1.0);

    dm_ptr lhr_ptr = to_ptr(lhr);
    dm_ptr rhr_ptr = to_ptr(rhr);

    ASSERT_EQ(dm_write(lhr_ptr, data.data(), 6), ok);
    ASSERT_EQ(dm_write(rhr_ptr, data.data() + 6, 6), ok);

    dm_ptr obj = nullptr;
    ASSERT_EQ(dm_mul(lhr_ptr, rhr_ptr, &obj), ok);
    ASSERT_NE(obj, nullptr);

    std::array<double, 4> actual {};
    ASSERT_EQ(dm_read(obj, actual.data(), actual.size()), ok);

    const std::array<double, 4> expected { 58.0, 64.0, 139.0, 154.0 };
    EXPECT_EQ(actual, expected);

    Java_dm_DenseMatrixJni_nativeDelete(nullptr, nullptr, to_obj(obj));
    Java_dm_DenseMatrixJni_nativeDelete(nullptr, nullptr, lhr);
    Java_dm_DenseMatrixJni_nativeDelete(nullptr, nullptr, rhr);
}