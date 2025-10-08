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
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("api")
public final class DenseMatrixApiJavaTest {
    private interface MatrixApi extends Library {
        MatrixApi INSTANCE = Native.load("matrix", MatrixApi.class);

        Pointer dm_new(long rows, long cols);
        void dm_delete(Pointer obj);
        long dm_rows(Pointer obj);
        long dm_cols(Pointer obj);
        long dm_size(Pointer obj);
        int dm_write(Pointer obj, double[] src, long valueCount);
        int dm_read(Pointer obj, double[] dst, long valueCount);
        int dm_mul(Pointer lhs, Pointer rhs, PointerByReference outObj);
    }

    private static final int STATUS_OK = 0;

    @Test
    void createMatrixTest() {
        Pointer obj = MatrixApi.INSTANCE.dm_new(2, 3);
        assertNotNull(obj, "obj must not be null");

        try {
            long rows = MatrixApi.INSTANCE.dm_rows(obj);
            long cols = MatrixApi.INSTANCE.dm_cols(obj);
            long size = MatrixApi.INSTANCE.dm_size(obj);

            assertEquals(2L, rows, "Unexpected row count");
            assertEquals(3L, cols, "Unexpected column count");
            assertEquals(6L, size, "Unexpected total element count");
        } finally {
            MatrixApi.INSTANCE.dm_delete(obj);
        }
    }

    @Test
    void multiplyTest() {
        Pointer lhs = MatrixApi.INSTANCE.dm_new(2, 3);
        Pointer rhs = MatrixApi.INSTANCE.dm_new(3, 2);
        assertNotNull(lhs, "lhs must not be null");
        assertNotNull(rhs, "rhs must not be null");

        double[] leftValues = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        double[] rightValues = { 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

        Pointer product = null;

        try {
            assertEquals(STATUS_OK, MatrixApi.INSTANCE.dm_write(lhs, leftValues, leftValues.length));
            assertEquals(STATUS_OK, MatrixApi.INSTANCE.dm_write(rhs, rightValues, rightValues.length));

            PointerByReference out = new PointerByReference();
            int status = MatrixApi.INSTANCE.dm_mul(lhs, rhs, out);
            assertEquals(STATUS_OK, status, "dm_mul should succeed for compatible dimensions");

            product = out.getValue();
            assertNotNull(product, "Product matrix should not be null");

            double[] actual = new double[4];
            assertEquals(STATUS_OK, MatrixApi.INSTANCE.dm_read(product, actual, actual.length));

            double[] expected = { 58.0, 64.0, 139.0, 154.0 };
            assertArrayEquals(expected, actual, "Matrix multiplication result is incorrect");
        } finally {
            if (product != null) {
                MatrixApi.INSTANCE.dm_delete(product);
            }
            MatrixApi.INSTANCE.dm_delete(rhs);
            MatrixApi.INSTANCE.dm_delete(lhs);
        }
    }
}
