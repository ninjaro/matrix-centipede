/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Yaroslav Riabtsev
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
#ifndef MATRIX_CENTIPEDE_DENSE_MATRIX_JNI_H
#define MATRIX_CENTIPEDE_DENSE_MATRIX_JNI_H

#include "dense_matrix_api.h"
#include <jni.h>

/**
 * @brief Utility macro that expands to the JNI export signature.
 */
#define wrap(type) JNIEXPORT type JNICALL

extern "C" {

/**
 * @brief Allocates a native matrix and returns its handle to Java.
 */
wrap(jlong) Java_dm_DenseMatrixJni_nativeNew(
    JNIEnv* env, jclass c, jlong row_count, jlong col_count
) noexcept;
/**
 * @brief Releases the matrix associated with @p obj.
 */
wrap(void) Java_dm_DenseMatrixJni_nativeDelete(
    JNIEnv* env, jclass c, jlong obj
) noexcept;
/**
 * @brief Returns the number of rows held by the native matrix.
 */
wrap(jlong) Java_dm_DenseMatrixJni_nativeRows(
    JNIEnv* env, jclass c, jlong obj
) noexcept;
/**
 * @brief Returns the number of columns held by the native matrix.
 */
wrap(jlong) Java_dm_DenseMatrixJni_nativeCols(
    JNIEnv* env, jclass c, jlong obj
) noexcept;
/**
 * @brief Returns the total number of elements stored by the matrix.
 */
wrap(jlong) Java_dm_DenseMatrixJni_nativeSize(
    JNIEnv* env, jclass c, jlong obj
) noexcept;

/**
 * @brief Copies @p value_count entries from @p src into the matrix buffer.
 */
wrap(jint) Java_dm_DenseMatrixJni_nativeWrite(
    JNIEnv* env, jclass c, jlong obj, jdoubleArray src, jlong value_count
) noexcept;

/**
 * @brief Reads matrix data into the provided Java array.
 */
wrap(jint) Java_dm_DenseMatrixJni_nativeRead(
    JNIEnv* env, jclass c, jlong obj, jdoubleArray dst, jlong value_count
) noexcept;

/**
 * @brief Multiplies two native matrices and stores the resulting handle.
 */
wrap(jint) Java_dm_DenseMatrixJni_nativeMul(
    JNIEnv* env, jclass c, jlong lhs, jlong rhs, jlongArray out_obj
) noexcept;

} // extern "C"

#endif // MATRIX_CENTIPEDE_DENSE_MATRIX_JNI_H
