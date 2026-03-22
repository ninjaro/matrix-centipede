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
package dm

import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Pointer
import com.sun.jna.ptr.PointerByReference
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull

@Tag("api")
class DenseMatrixApiKotlinTest {
    private interface MatrixApi : Library {
        fun dm_new_empty(): Pointer?
        fun dm_new(rows: Long, cols: Long): Pointer?
        fun dm_delete(obj: Pointer?)
        fun dm_rows(obj: Pointer?): Long
        fun dm_cols(obj: Pointer?): Long
        fun dm_size(obj: Pointer?): Long
        fun dm_write(obj: Pointer?, src: DoubleArray?, valueCount: Long): Int
        fun dm_read(obj: Pointer?, dst: DoubleArray?, valueCount: Long): Int
        fun dm_mul(lhs: Pointer?, rhs: Pointer?, outObj: PointerByReference?): Int
    }

    private companion object {
        private val matrixApiLib: MatrixApi = Native.load("matrix", MatrixApi::class.java)
        private const val STATUS_OK = 0
        private const val STATUS_NULL = 1
        private const val STATUS_BAD_SIZE = 2
    }

    @Test
    fun createMatrixTest() {
        val obj = matrixApiLib.dm_new(2, 3)
        assertNotNull(obj)

        try {
            val rows = matrixApiLib.dm_rows(obj)
            val cols = matrixApiLib.dm_cols(obj)
            val size = matrixApiLib.dm_size(obj)

            assertEquals(2L, rows)
            assertEquals(3L, cols)
            assertEquals(6L, size)
        } finally {
            matrixApiLib.dm_delete(obj)
        }
    }

    @Test
    fun multiplyTest() {
        val lhs = matrixApiLib.dm_new(2, 3)
        val rhs = matrixApiLib.dm_new(3, 2)
        assertNotNull(lhs)
        assertNotNull(rhs)

        val leftValues = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val rightValues = doubleArrayOf(7.0, 8.0, 9.0, 10.0, 11.0, 12.0)

        var product: Pointer? = null

        try {
            assertEquals(STATUS_OK, matrixApiLib.dm_write(lhs, leftValues, leftValues.size.toLong()))
            assertEquals(STATUS_OK, matrixApiLib.dm_write(rhs, rightValues, rightValues.size.toLong()))

            val out = PointerByReference()
            val status = matrixApiLib.dm_mul(lhs, rhs, out)
            assertEquals(STATUS_OK, status)

            product = out.value
            assertNotNull(product)

            val actual = DoubleArray(4)
            assertEquals(STATUS_OK, matrixApiLib.dm_read(product, actual, actual.size.toLong()))

            val expected = doubleArrayOf(58.0, 64.0, 139.0, 154.0)
            assertArrayEquals(expected, actual)
        } finally {
            if (product != null) {
                matrixApiLib.dm_delete(product)
            }
            matrixApiLib.dm_delete(rhs)
            matrixApiLib.dm_delete(lhs)
        }
    }

    @Test
    fun returnZeroTest() {
        assertEquals(0L, matrixApiLib.dm_rows(Pointer.NULL))
        assertEquals(0L, matrixApiLib.dm_cols(Pointer.NULL))
        assertEquals(0L, matrixApiLib.dm_size(Pointer.NULL))
    }

    @Test
    fun newEmptyTest() {
        val obj = matrixApiLib.dm_new_empty()
        assertNotNull(obj)
        matrixApiLib.dm_delete(obj)
    }

    @Test
    fun readWriteNullTest() {
        val obj = matrixApiLib.dm_new(1, 3)
        assertNotNull(obj)

        try {
            assertEquals(STATUS_NULL, matrixApiLib.dm_write(obj, null, 3))
            assertEquals(STATUS_NULL, matrixApiLib.dm_read(obj, null, 3))
        } finally {
            matrixApiLib.dm_delete(obj)
        }

        val nobj = matrixApiLib.dm_new(1, 0)
        assertNotNull(nobj)

        try {
            assertEquals(STATUS_OK, matrixApiLib.dm_write(nobj, null, 0))
            assertEquals(STATUS_OK, matrixApiLib.dm_read(nobj, null, 0))
        } finally {
            matrixApiLib.dm_delete(nobj)
        }

        assertEquals(STATUS_NULL, matrixApiLib.dm_read(Pointer.NULL, null, 0))

        val buffer: DoubleArray? = null
        assertEquals(STATUS_NULL, matrixApiLib.dm_write(Pointer.NULL, buffer, 3))
        assertEquals(STATUS_NULL, matrixApiLib.dm_read(Pointer.NULL, buffer, 3))
    }

    @Test
    fun readWriteBadSizeTest() {
        val obj = matrixApiLib.dm_new(2, 2)
        assertNotNull(obj)

        val buffer = DoubleArray(6)

        try {
            assertEquals(STATUS_BAD_SIZE, matrixApiLib.dm_write(obj, buffer, 5))
            assertEquals(STATUS_BAD_SIZE, matrixApiLib.dm_read(obj, buffer, 3))
        } finally {
            matrixApiLib.dm_delete(obj)
        }
    }

    @Test
    fun badMulTest() {
        val a = matrixApiLib.dm_new(2, 3)
        val b = matrixApiLib.dm_new(4, 5)
        assertNotNull(a)
        assertNotNull(b)

        try {
            val outRef = PointerByReference()
            assertEquals(STATUS_NULL, matrixApiLib.dm_mul(Pointer.NULL, b, outRef))
            assertEquals(STATUS_NULL, matrixApiLib.dm_mul(a, Pointer.NULL, outRef))
            assertEquals(STATUS_NULL, matrixApiLib.dm_mul(a, b, null))

            val mismatchOut = PointerByReference()
            assertEquals(STATUS_BAD_SIZE, matrixApiLib.dm_mul(a, b, mismatchOut))
            assertNull(mismatchOut.value)
        } finally {
            matrixApiLib.dm_delete(b)
            matrixApiLib.dm_delete(a)
        }
    }

    @Test
    fun deleteNullTest() {
        matrixApiLib.dm_delete(Pointer.NULL)
    }

    @Test
    fun overflowTest() {
        val maxN = -1L

        val a = matrixApiLib.dm_new(maxN, 0)
        val b = matrixApiLib.dm_new(0, maxN)
        assertNotNull(a)
        assertNotNull(b)

        try {
            val outRef = PointerByReference()
            assertEquals(STATUS_BAD_SIZE, matrixApiLib.dm_mul(a, b, outRef))
            assertNull(outRef.value)
        } finally {
            matrixApiLib.dm_delete(b)
            matrixApiLib.dm_delete(a)
        }
    }
}
