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
#include "dense_matrix_jni.h"

#include <cstdint>

[[nodiscard]] dm_ptr from_handle(jlong obj) noexcept {
    if (obj == 0) {
        return nullptr;
    }
    return reinterpret_cast<dm_ptr>(static_cast<std::uintptr_t>(obj));
}

[[nodiscard]] jlong to_handle(dm_ptr ptr) noexcept {
    return static_cast<jlong>(reinterpret_cast<std::uintptr_t>(ptr));
}

[[nodiscard]] jint to_jint(dm_status status) noexcept {
    return static_cast<jint>(status);
}

[[nodiscard]] bool
fits_in_array(JNIEnv* env, jarray array, jlong count) noexcept {
    if (array == nullptr) {
        return false;
    }
    const jsize length = env->GetArrayLength(array);
    return static_cast<jlong>(length) >= count;
}

extern "C" {

wrap(jlong) Java_dm_DenseMatrixJni_nativeNew(
    JNIEnv* env, jclass, jlong row_count, jlong col_count
) noexcept {
    if (row_count < 0 || col_count < 0) {
        return 0;
    }
    (void)env;
    dm_ptr obj = dm_new(
        static_cast<size_t>(row_count), static_cast<size_t>(col_count)
    );
    return to_handle(obj);
}

wrap(void) Java_dm_DenseMatrixJni_nativeDelete(
    JNIEnv*, jclass, jlong obj
) noexcept {
    dm_delete(from_handle(obj));
}

wrap(jlong) Java_dm_DenseMatrixJni_nativeRows(
    JNIEnv*, jclass, jlong obj
) noexcept {
    return static_cast<jlong>(dm_rows(from_handle(obj)));
}

wrap(jlong) Java_dm_DenseMatrixJni_nativeCols(
    JNIEnv*, jclass, jlong obj
) noexcept {
    return static_cast<jlong>(dm_cols(from_handle(obj)));
}

wrap(jlong) Java_dm_DenseMatrixJni_nativeSize(
    JNIEnv*, jclass, jlong obj
) noexcept {
    return static_cast<jlong>(dm_size(from_handle(obj)));
}

wrap(jint) Java_dm_DenseMatrixJni_nativeWrite(
    JNIEnv* env, jclass, jlong obj, jdoubleArray src, jlong value_count
) noexcept {
    if (value_count < 0) {
        return to_jint(bad_size);
    }
    dm_ptr ptr = from_handle(obj);
    const size_t count = static_cast<size_t>(value_count);

    if (count == 0) {
        return to_jint(dm_write(ptr, nullptr, 0));
    }
    if (src == nullptr) {
        return to_jint(null);
    }
    if (!fits_in_array(env, src, value_count)) {
        return to_jint(bad_size);
    }

    jdouble* elements = env->GetDoubleArrayElements(src, nullptr);
    if (elements == nullptr) {
        return to_jint(bad_alloc);
    }

    dm_status status
        = dm_write(ptr, reinterpret_cast<const double*>(elements), count);
    env->ReleaseDoubleArrayElements(src, elements, JNI_ABORT);
    return to_jint(status);
}

wrap(jint) Java_dm_DenseMatrixJni_nativeRead(
    JNIEnv* env, jclass, jlong obj, jdoubleArray dst, jlong value_count
) noexcept {
    if (value_count < 0) {
        return to_jint(bad_size);
    }
    dm_ptr ptr = from_handle(obj);
    const size_t count = static_cast<size_t>(value_count);

    if (count == 0) {
        return to_jint(dm_read(ptr, nullptr, 0));
    }
    if (dst == nullptr) {
        return to_jint(null);
    }
    if (!fits_in_array(env, dst, value_count)) {
        return to_jint(bad_size);
    }

    jdouble* elements = env->GetDoubleArrayElements(dst, nullptr);
    if (elements == nullptr) {
        return to_jint(bad_alloc);
    }

    dm_status status = dm_read(ptr, reinterpret_cast<double*>(elements), count);
    const jint mode = (status == ok) ? 0 : JNI_ABORT;
    env->ReleaseDoubleArrayElements(dst, elements, mode);
    return to_jint(status);
}

wrap(jint) Java_dm_DenseMatrixJni_nativeMul(
    JNIEnv* env, jclass, jlong lhs, jlong rhs, jlongArray out_obj
) noexcept {
    if (out_obj == nullptr) {
        return to_jint(null);
    }
    if (!fits_in_array(env, out_obj, 1)) {
        return to_jint(bad_size);
    }

    dm_ptr result = nullptr;
    dm_status status = dm_mul(from_handle(lhs), from_handle(rhs), &result);

    jlong handle = (status == ok) ? to_handle(result) : 0;
    env->SetLongArrayRegion(out_obj, 0, 1, &handle);
    return to_jint(status);
}

} // extern "C"
