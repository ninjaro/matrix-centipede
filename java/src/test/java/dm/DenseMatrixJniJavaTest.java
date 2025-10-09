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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("jni")
public final class DenseMatrixJniJavaTest {
    private static final int STATUS_OK = 0;
    private static final int STATUS_NULL = 1;
    private static final int STATUS_BAD_SIZE = 2;


    @Test
    void createMatrixTest() {
        long matrix = DenseMatrixJni.newMatrix(2, 3);
        assertNotEquals(0L, matrix);

        try {
            assertEquals(2L, DenseMatrixJni.rows(matrix));
            assertEquals(3L, DenseMatrixJni.cols(matrix));
            assertEquals(6L, DenseMatrixJni.size(matrix));
        } finally {
            DenseMatrixJni.delete(matrix);
        }
    }

    @Test
    void multiplyTest() {
        long lhs = DenseMatrixJni.newMatrix(2, 3);
        long rhs = DenseMatrixJni.newMatrix(3, 2);
        assertNotEquals(0L, lhs);
        assertNotEquals(0L, rhs);

        double[] leftValues = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        double[] rightValues = { 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

        long product = 0L;

        try {
            assertEquals(STATUS_OK, DenseMatrixJni.write(lhs, leftValues, 6));
            assertEquals(STATUS_OK, DenseMatrixJni.write(rhs, rightValues, 6));

            long[] productHandle = new long[1];
            int status = DenseMatrixJni.mul(lhs, rhs, productHandle);
            assertEquals(STATUS_OK, status);

            product = productHandle[0];
            assertNotEquals(0L, product);

            double[] actual = new double[4];
            assertEquals(STATUS_OK, DenseMatrixJni.read(product, actual, 4));

            double[] expected = { 58.0, 64.0, 139.0, 154.0 };
            assertArrayEquals(expected, actual);
        } finally {
            if (product != 0L) {
                DenseMatrixJni.delete(product);
            }
            DenseMatrixJni.delete(rhs);
            DenseMatrixJni.delete(lhs);
        }
    }

    @Test
    void returnZeroTest() {
        assertEquals(0L, DenseMatrixJni.rows(0L));
        assertEquals(0L, DenseMatrixJni.cols(0L));
        assertEquals(0L, DenseMatrixJni.size(0L));
    }

    @Test
    void newEmptyTest() {
        long handle = DenseMatrixJni.newEmpty();
        assertNotEquals(0L, handle);

        assertDoesNotThrow(() -> DenseMatrixJni.delete(handle));
    }

    @Test
    void readWriteNullTest() {
        long handle = DenseMatrixJni.newMatrix(1, 3);
        assertNotEquals(0L, handle);

        try {
            assertEquals(STATUS_NULL, DenseMatrixJni.write(handle, null, 3));
            assertEquals(STATUS_NULL, DenseMatrixJni.read(handle, null, 3));
        } finally {
            DenseMatrixJni.delete(handle);
        }

        long zeroSized = DenseMatrixJni.newMatrix(1, 0);
        assertNotEquals(0L, zeroSized);

        try {
            assertEquals(STATUS_OK, DenseMatrixJni.write(zeroSized, null, 0));
            assertEquals(STATUS_OK, DenseMatrixJni.read(zeroSized, null, 0));
        } finally {
            DenseMatrixJni.delete(zeroSized);
        }

        assertEquals(STATUS_NULL, DenseMatrixJni.read(0L, null, 0));
        assertEquals(STATUS_NULL, DenseMatrixJni.write(0L, null, 3));
        assertEquals(STATUS_NULL, DenseMatrixJni.read(0L, null, 3));
    }

    @Test
    void readWriteBadSizeTest() {
        long handle = DenseMatrixJni.newMatrix(2, 2);
        assertNotEquals(0L, handle);

        try {
            double[] buffer = new double[6];
            assertEquals(STATUS_BAD_SIZE, DenseMatrixJni.write(handle, buffer, 5));
            assertEquals(STATUS_BAD_SIZE, DenseMatrixJni.read(handle, buffer, 3));
        } finally {
            DenseMatrixJni.delete(handle);
        }
    }

    @Test
    void badMulTest() {
        long lhs = DenseMatrixJni.newMatrix(2, 3);
        long rhs = DenseMatrixJni.newMatrix(4, 5);
        assertNotEquals(0L, lhs);
        assertNotEquals(0L, rhs);

        try {
            long[] out = new long[1];

            assertEquals(STATUS_NULL, DenseMatrixJni.mul(0L, rhs, out));
            assertEquals(STATUS_NULL, DenseMatrixJni.mul(lhs, 0L, out));
            assertEquals(STATUS_NULL, DenseMatrixJni.mul(lhs, rhs, null));

            assertEquals(STATUS_BAD_SIZE, DenseMatrixJni.mul(lhs, rhs, out));
            assertEquals(0L, out[0]);
        } finally {
            DenseMatrixJni.delete(rhs);
            DenseMatrixJni.delete(lhs);
        }
    }

    @Test
    void deleteNullTest() {
        assertDoesNotThrow(() -> DenseMatrixJni.delete(0L));
    }

    @Test
    void overflowTest() {
        long lhs = DenseMatrixJni.newMatrix(Long.MAX_VALUE, 0L);
        long rhs = DenseMatrixJni.newMatrix(0L, Long.MAX_VALUE);
        assertNotEquals(0L, lhs);
        assertNotEquals(0L, rhs);

        try {
            long[] out = new long[1];
            assertEquals(STATUS_BAD_SIZE, DenseMatrixJni.mul(lhs, rhs, out));
            assertEquals(0L, out[0]);
        } finally {
            DenseMatrixJni.delete(rhs);
            DenseMatrixJni.delete(lhs);
        }
    }
}