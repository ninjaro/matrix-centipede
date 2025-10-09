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
import static org.junit.jupiter.api.Assertions.assertNull;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("api")
public final class DenseMatrixApiJavaTest {
    private interface MatrixApi extends Library {
        MatrixApi MatrixApiLib = Native.load("matrix", MatrixApi.class);

        Pointer dm_new_empty();
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
    private static final int STATUS_NULL = 1;
    private static final int STATUS_BAD_SIZE = 2;

    @Test
    void createMatrixTest() {
        Pointer obj = MatrixApi.MatrixApiLib.dm_new(2, 3);
        assertNotNull(obj);

        try {
            long rows = MatrixApi.MatrixApiLib.dm_rows(obj);
            long cols = MatrixApi.MatrixApiLib.dm_cols(obj);
            long size = MatrixApi.MatrixApiLib.dm_size(obj);

            assertEquals(2L, rows);
            assertEquals(3L, cols);
            assertEquals(6L, size);
        } finally {
            MatrixApi.MatrixApiLib.dm_delete(obj);
        }
    }

    @Test
    void multiplyTest() {
        Pointer lhs = MatrixApi.MatrixApiLib.dm_new(2, 3);
        Pointer rhs = MatrixApi.MatrixApiLib.dm_new(3, 2);
        assertNotNull(lhs);
        assertNotNull(rhs);

        double[] leftValues = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        double[] rightValues = { 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

        Pointer result = null;

        try {
            assertEquals(STATUS_OK, MatrixApi.MatrixApiLib.dm_write(lhs, leftValues, leftValues.length));
            assertEquals(STATUS_OK, MatrixApi.MatrixApiLib.dm_write(rhs, rightValues, rightValues.length));

            PointerByReference out = new PointerByReference();
            int status = MatrixApi.MatrixApiLib.dm_mul(lhs, rhs, out);
            assertEquals(STATUS_OK, status);

            result = out.getValue();
            assertNotNull(result);

            double[] actual = new double[4];
            assertEquals(STATUS_OK, MatrixApi.MatrixApiLib.dm_read(result, actual, actual.length));

            double[] expected = { 58.0, 64.0, 139.0, 154.0 };
            assertArrayEquals(expected, actual);
        } finally {
            if (result != null) {
                MatrixApi.MatrixApiLib.dm_delete(result);
            }
            MatrixApi.MatrixApiLib.dm_delete(rhs);
            MatrixApi.MatrixApiLib.dm_delete(lhs);
        }
    }

    @Test
    void returnZeroTest() {
        assertEquals(0L, MatrixApi.MatrixApiLib.dm_rows(Pointer.NULL));
        assertEquals(0L, MatrixApi.MatrixApiLib.dm_cols(Pointer.NULL));
        assertEquals(0L, MatrixApi.MatrixApiLib.dm_size(Pointer.NULL));
    }

    @Test
    void newEmptyTest() {
        Pointer obj = MatrixApi.MatrixApiLib.dm_new_empty();
        assertNotNull(obj);
        MatrixApi.MatrixApiLib.dm_delete(obj);
    }

    @Test
    void readWriteNullTest() {
        Pointer obj = MatrixApi.MatrixApiLib.dm_new(1, 3);
        assertNotNull(obj);

        try {
            assertEquals(STATUS_NULL, MatrixApi.MatrixApiLib.dm_write(obj, null, 3));
            assertEquals(STATUS_NULL, MatrixApi.MatrixApiLib.dm_read(obj, null, 3));
        } finally {
            MatrixApi.MatrixApiLib.dm_delete(obj);
        }

        Pointer nobj = MatrixApi.MatrixApiLib.dm_new(1, 0);
        assertNotNull(nobj);

        try {
            assertEquals(STATUS_OK, MatrixApi.MatrixApiLib.dm_write(nobj, null, 0));
            assertEquals(STATUS_OK, MatrixApi.MatrixApiLib.dm_read(nobj, null, 0));
        } finally {
            MatrixApi.MatrixApiLib.dm_delete(nobj);
        }

        assertEquals(STATUS_NULL, MatrixApi.MatrixApiLib.dm_read(Pointer.NULL, null, 0));

        double[] buffer = null;
        assertEquals(STATUS_NULL, MatrixApi.MatrixApiLib.dm_write(Pointer.NULL, buffer, 3));
        assertEquals(STATUS_NULL, MatrixApi.MatrixApiLib.dm_read(Pointer.NULL, buffer, 3));
    }

    @Test
    void readWriteBadSizeTest() {
        Pointer obj = MatrixApi.MatrixApiLib.dm_new(2, 2);
        assertNotNull(obj);

        double[] buffer = new double[6];

        try {
            assertEquals(STATUS_BAD_SIZE, MatrixApi.MatrixApiLib.dm_write(obj, buffer, 5));
            assertEquals(STATUS_BAD_SIZE, MatrixApi.MatrixApiLib.dm_read(obj, buffer, 3));
        } finally {
            MatrixApi.MatrixApiLib.dm_delete(obj);
        }
    }

    @Test
    void badMulTest() {
        Pointer a = MatrixApi.MatrixApiLib.dm_new(2, 3);
        Pointer b = MatrixApi.MatrixApiLib.dm_new(4, 5);
        assertNotNull(a);
        assertNotNull(b);

        try {
            PointerByReference outRef = new PointerByReference();
            assertEquals(STATUS_NULL, MatrixApi.MatrixApiLib.dm_mul(Pointer.NULL, b, outRef));
            assertEquals(STATUS_NULL, MatrixApi.MatrixApiLib.dm_mul(a, Pointer.NULL, outRef));
            assertEquals(STATUS_NULL, MatrixApi.MatrixApiLib.dm_mul(a, b, null));

            PointerByReference mismatchOut = new PointerByReference();
            assertEquals(STATUS_BAD_SIZE, MatrixApi.MatrixApiLib.dm_mul(a, b, mismatchOut));
            assertNull(mismatchOut.getValue());
        } finally {
            MatrixApi.MatrixApiLib.dm_delete(b);
            MatrixApi.MatrixApiLib.dm_delete(a);
        }
    }

    @Test
    void deleteNullTest() {
        MatrixApi.MatrixApiLib.dm_delete(Pointer.NULL);
    }

    @Test
    void overflowTest() {
        long maxN = -1L;

        Pointer a = MatrixApi.MatrixApiLib.dm_new(maxN, 0);
        Pointer b = MatrixApi.MatrixApiLib.dm_new(0, maxN);
        assertNotNull(a);
        assertNotNull(b);

        try {
            PointerByReference outRef = new PointerByReference();
            assertEquals(STATUS_BAD_SIZE, MatrixApi.MatrixApiLib.dm_mul(a, b, outRef));
            assertNull(outRef.getValue());
        } finally {
            MatrixApi.MatrixApiLib.dm_delete(b);
            MatrixApi.MatrixApiLib.dm_delete(a);
        }
    }
}