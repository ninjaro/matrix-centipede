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
package dm;

final class DenseMatrixJni {
    static {
        System.loadLibrary("matrix_jni");
    }

    private DenseMatrixJni() {
        // Utility class
    }

    private static native long nativeNewEmpty();
    private static native long nativeNew(long rows, long cols);
    private static native void nativeDelete(long handle);
    private static native long nativeRows(long handle);
    private static native long nativeCols(long handle);
    private static native long nativeSize(long handle);
    private static native int nativeWrite(long handle, double[] src, long valueCount);
    private static native int nativeRead(long handle, double[] dst, long valueCount);
    private static native int nativeMul(long lhs, long rhs, long[] outHandle);

    static long newEmpty() {
        return nativeNewEmpty();
    }

    static long newMatrix(long rows, long cols) {
        return nativeNew(rows, cols);
    }

    static void delete(long handle) {
        nativeDelete(handle);
    }

    static long rows(long handle) {
        return nativeRows(handle);
    }

    static long cols(long handle) {
        return nativeCols(handle);
    }

    static long size(long handle) {
        return nativeSize(handle);
    }

    static int write(long handle, double[] values, long length) {
        return nativeWrite(handle, values, length);
    }

    static int read(long handle, double[] values, long length) {
        return nativeRead(handle, values, length);
    }

    static int mul(long lhs, long rhs, long[] outHandle) {
        return nativeMul(lhs, rhs, outHandle);
    }
}