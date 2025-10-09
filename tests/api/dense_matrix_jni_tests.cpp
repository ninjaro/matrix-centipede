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

class jvm_handle {
public:
    jvm_handle() {
        JavaVMInitArgs args {};
        args.version = JNI_VERSION_1_8;
        args.options = nullptr;
        args.nOptions = 0;
        args.ignoreUnrecognized = JNI_TRUE;

        jint rc
            = JNI_CreateJavaVM(&vm_, reinterpret_cast<void**>(&env_), &args);
        if (rc != JNI_OK) {
            throw std::runtime_error("failed to create Java VM for tests");
        }
    }

    jvm_handle(const jvm_handle&) = delete;
    jvm_handle& operator=(const jvm_handle&) = delete;

    ~jvm_handle() {
        if (vm_ != nullptr) {
            vm_->DestroyJavaVM();
        }
    }

    [[nodiscard]] JNIEnv* env() const noexcept { return env_; }

private:
    JavaVM* vm_ = nullptr;
    JNIEnv* env_ = nullptr;
};

JNIEnv* get_env() {
    static jvm_handle instance;
    return instance.env();
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
    JNIEnv* env = get_env();

    jlong lhr = Java_dm_DenseMatrixJni_nativeNew(nullptr, nullptr, 2, 3);
    jlong rhr = Java_dm_DenseMatrixJni_nativeNew(nullptr, nullptr, 3, 2);
    ASSERT_NE(lhr, 0);
    ASSERT_NE(rhr, 0);

    const jdouble lhr_data[6] = { 1, 2, 3, 4, 5, 6 };
    const jdouble rhr_data[6] = { 7, 8, 9, 10, 11, 12 };

    jdoubleArray lhr_arr = env->NewDoubleArray(6);
    jdoubleArray rhr_arr = env->NewDoubleArray(6);
    ASSERT_NE(lhr_arr, nullptr);
    ASSERT_NE(rhr_arr, nullptr);
    env->SetDoubleArrayRegion(lhr_arr, 0, 6, lhr_data);
    env->SetDoubleArrayRegion(rhr_arr, 0, 6, rhr_data);

    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeWrite(env, nullptr, lhr, lhr_arr, 6),
        static_cast<jint>(ok)
    );
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeWrite(env, nullptr, rhr, rhr_arr, 6),
        static_cast<jint>(ok)
    );

    jlongArray out = env->NewLongArray(1);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeMul(env, nullptr, lhr, rhr, out),
        static_cast<jint>(ok)
    );

    jlong c = 0;
    env->GetLongArrayRegion(out, 0, 1, &c);
    ASSERT_NE(c, 0);

    EXPECT_EQ(Java_dm_DenseMatrixJni_nativeRows(env, nullptr, c), 2);
    EXPECT_EQ(Java_dm_DenseMatrixJni_nativeCols(env, nullptr, c), 2);
    EXPECT_EQ(Java_dm_DenseMatrixJni_nativeSize(env, nullptr, c), 4);

    jdoubleArray Carr = env->NewDoubleArray(4);
    ASSERT_NE(Carr, nullptr);
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeRead(env, nullptr, c, Carr, 4),
        static_cast<jint>(ok)
    );
    jdouble Cdata[4] = { 0, 0, 0, 0 };
    env->GetDoubleArrayRegion(Carr, 0, 4, Cdata);

    const double expected[4] = { 58.0, 64.0, 139.0, 154.0 };
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(Cdata[i], expected[i]);
    }

    Java_dm_DenseMatrixJni_nativeDelete(env, nullptr, lhr);
    Java_dm_DenseMatrixJni_nativeDelete(env, nullptr, rhr);
    Java_dm_DenseMatrixJni_nativeDelete(env, nullptr, c);
    env->DeleteLocalRef(lhr_arr);
    env->DeleteLocalRef(rhr_arr);
    env->DeleteLocalRef(Carr);
    env->DeleteLocalRef(out);
}

TEST(DenseMatrixJniTest, ReturnZeroTest) {
    const jlong nil = 0;
    EXPECT_EQ(Java_dm_DenseMatrixJni_nativeRows(nullptr, nullptr, nil), 0);
    EXPECT_EQ(Java_dm_DenseMatrixJni_nativeCols(nullptr, nullptr, nil), 0);
    EXPECT_EQ(Java_dm_DenseMatrixJni_nativeSize(nullptr, nullptr, nil), 0);
}

TEST(DenseMatrixJniTest, NewEmptyTest) {
    jlong h = Java_dm_DenseMatrixJni_nativeNewEmpty(nullptr, nullptr);
    ASSERT_NE(h, 0);
    EXPECT_NO_THROW(Java_dm_DenseMatrixJni_nativeDelete(nullptr, nullptr, h));
}

TEST(DenseMatrixJniTest, ReadWriteNullTest) {
    jlong obj = Java_dm_DenseMatrixJni_nativeNew(nullptr, nullptr, 1, 3);
    ASSERT_NE(obj, 0);
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeWrite(nullptr, nullptr, obj, nullptr, 3),
        static_cast<jint>(null)
    );
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeRead(nullptr, nullptr, obj, nullptr, 3),
        static_cast<jint>(null)
    );
    Java_dm_DenseMatrixJni_nativeDelete(nullptr, nullptr, obj);

    jlong nobj = Java_dm_DenseMatrixJni_nativeNew(nullptr, nullptr, 1, 0);
    ASSERT_NE(nobj, 0);
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeWrite(nullptr, nullptr, nobj, nullptr, 0),
        static_cast<jint>(ok)
    );
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeRead(nullptr, nullptr, nobj, nullptr, 0),
        static_cast<jint>(ok)
    );
    Java_dm_DenseMatrixJni_nativeDelete(nullptr, nullptr, nobj);

    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeRead(nullptr, nullptr, 0, nullptr, 0),
        static_cast<jint>(null)
    );
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeWrite(nullptr, nullptr, 0, nullptr, 3),
        static_cast<jint>(null)
    );
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeRead(nullptr, nullptr, 0, nullptr, 3),
        static_cast<jint>(null)
    );
}

TEST(DenseMatrixJniTest, ReadWriteBadSizeTest) {
    JNIEnv* env = get_env();
    jlong obj = Java_dm_DenseMatrixJni_nativeNew(env, nullptr, 2, 2);
    ASSERT_NE(obj, 0);

    jdoubleArray buf = env->NewDoubleArray(6);
    ASSERT_NE(buf, nullptr);

    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeWrite(env, nullptr, obj, buf, 5),
        static_cast<jint>(bad_size)
    );
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeRead(env, nullptr, obj, buf, 3),
        static_cast<jint>(bad_size)
    );

    Java_dm_DenseMatrixJni_nativeDelete(env, nullptr, obj);
    env->DeleteLocalRef(buf);
}

TEST(DenseMatrixJniTest, BadMulTest) {
    JNIEnv* env = get_env();

    jlong a = Java_dm_DenseMatrixJni_nativeNew(env, nullptr, 2, 3);
    jlong b = Java_dm_DenseMatrixJni_nativeNew(env, nullptr, 4, 5);
    ASSERT_NE(a, 0);
    ASSERT_NE(b, 0);

    jlongArray out = env->NewLongArray(1);
    ASSERT_NE(out, nullptr);

    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeMul(nullptr, nullptr, 0, b, out),
        static_cast<jint>(null)
    );
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeMul(nullptr, nullptr, a, 0, out),
        static_cast<jint>(null)
    );
    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeMul(env, nullptr, a, b, nullptr),
        static_cast<jint>(null)
    );

    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeMul(env, nullptr, a, b, out),
        static_cast<jint>(bad_size)
    );
    jlong out_handle = -1;
    env->GetLongArrayRegion(out, 0, 1, &out_handle);
    EXPECT_EQ(out_handle, 0);

    Java_dm_DenseMatrixJni_nativeDelete(env, nullptr, a);
    Java_dm_DenseMatrixJni_nativeDelete(env, nullptr, b);
    env->DeleteLocalRef(out);
}

TEST(DenseMatrixJniTest, DeleteNullTest) {
    EXPECT_NO_THROW(Java_dm_DenseMatrixJni_nativeDelete(nullptr, nullptr, 0));
}

TEST(DenseMatrixJniTest, OverflowTest) {
    JNIEnv* env = get_env();

    const jlong MAX = std::numeric_limits<jlong>::max();

    jlong a = Java_dm_DenseMatrixJni_nativeNew(env, nullptr, MAX, 0);
    jlong b = Java_dm_DenseMatrixJni_nativeNew(env, nullptr, 0, MAX);
    ASSERT_NE(a, 0);
    ASSERT_NE(b, 0);

    jlongArray out = env->NewLongArray(1);
    ASSERT_NE(out, nullptr);

    EXPECT_EQ(
        Java_dm_DenseMatrixJni_nativeMul(env, nullptr, a, b, out),
        static_cast<jint>(bad_size)
    );
    jlong out_handle = -1;
    env->GetLongArrayRegion(out, 0, 1, &out_handle);
    EXPECT_EQ(out_handle, 0);

    Java_dm_DenseMatrixJni_nativeDelete(env, nullptr, a);
    Java_dm_DenseMatrixJni_nativeDelete(env, nullptr, b);
    env->DeleteLocalRef(out);
}
