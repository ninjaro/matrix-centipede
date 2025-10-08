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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("jni")
public final class DenseMatrixJniTest {
    private static final int STATUS_OK = 0;

    @Test
    void createMatrixTest() {
        long matrix = DenseMatrixJni.newMatrix(2, 3);
        assertNotEquals(0L, matrix, "Matrix handle must not be zero");

        try {
            assertEquals(2L, DenseMatrixJni.rows(matrix), "Unexpected row count");
            assertEquals(3L, DenseMatrixJni.cols(matrix), "Unexpected column count");
            assertEquals(6L, DenseMatrixJni.size(matrix), "Unexpected element count");
        } finally {
            DenseMatrixJni.delete(matrix);
        }
    }

    @Test
    void multiplyTest() {
        long lhs = DenseMatrixJni.newMatrix(2, 3);
        long rhs = DenseMatrixJni.newMatrix(3, 2);
        assertNotEquals(0L, lhs, "Left-hand matrix handle must not be zero");
        assertNotEquals(0L, rhs, "Right-hand matrix handle must not be zero");

        double[] leftValues = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        double[] rightValues = { 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

        long product = 0L;

        try {
            assertEquals(STATUS_OK, DenseMatrixJni.write(lhs, leftValues));
            assertEquals(STATUS_OK, DenseMatrixJni.write(rhs, rightValues));

            long[] productHandle = new long[1];
            int status = DenseMatrixJni.mul(lhs, rhs, productHandle);
            assertEquals(STATUS_OK, status, "dm_mul should succeed for compatible matrices");

            product = productHandle[0];
            assertNotEquals(0L, product, "Product matrix handle must not be zero");

            double[] actual = new double[4];
            assertEquals(STATUS_OK, DenseMatrixJni.read(product, actual));

            double[] expected = { 58.0, 64.0, 139.0, 154.0 };
            assertArrayEquals(expected, actual, "Matrix multiplication result is incorrect");
        } finally {
            if (product != 0L) {
                DenseMatrixJni.delete(product);
            }
            DenseMatrixJni.delete(rhs);
            DenseMatrixJni.delete(lhs);
        }
    }
}