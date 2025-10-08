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

@Tag("api")
class DenseMatrixApiKotlinTest {
    private interface MatrixApi : Library {
        fun dm_new(rows: Long, cols: Long): Pointer?
        fun dm_delete(obj: Pointer?)
        fun dm_rows(obj: Pointer?): Long
        fun dm_cols(obj: Pointer?): Long
        fun dm_size(obj: Pointer?): Long
        fun dm_write(obj: Pointer?, src: DoubleArray, valueCount: Long): Int
        fun dm_read(obj: Pointer?, dst: DoubleArray, valueCount: Long): Int
        fun dm_mul(lhs: Pointer?, rhs: Pointer?, outObj: PointerByReference): Int
    }

    private companion object {
        private val matrixApi: MatrixApi = Native.load("matrix", MatrixApi::class.java)
        private const val STATUS_OK = 0
    }

    @Test
    fun createMatrixTest() {
        val obj = matrixApi.dm_new(2, 3)
        assertNotNull(obj, "obj must not be null")

        try {
            val rows = matrixApi.dm_rows(obj)
            val cols = matrixApi.dm_cols(obj)
            val size = matrixApi.dm_size(obj)

            assertEquals(2L, rows, "Unexpected row count")
            assertEquals(3L, cols, "Unexpected column count")
            assertEquals(6L, size, "Unexpected total element count")
        } finally {
            matrixApi.dm_delete(obj)
        }
    }

    @Test
    fun multiplyTest() {
        val lhs = matrixApi.dm_new(2, 3)
        val rhs = matrixApi.dm_new(3, 2)
        assertNotNull(lhs, "lhs must not be null")
        assertNotNull(rhs, "rhs must not be null")

        val leftValues = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val rightValues = doubleArrayOf(7.0, 8.0, 9.0, 10.0, 11.0, 12.0)

        var product: Pointer? = null

        try {
            assertEquals(STATUS_OK, matrixApi.dm_write(lhs, leftValues, leftValues.size.toLong()))
            assertEquals(STATUS_OK, matrixApi.dm_write(rhs, rightValues, rightValues.size.toLong()))

            val out = PointerByReference()
            val status = matrixApi.dm_mul(lhs, rhs, out)
            assertEquals(STATUS_OK, status, "dm_mul should succeed for compatible dimensions")

            product = out.value
            assertNotNull(product, "Product matrix should not be null")

            val actual = DoubleArray(4)
            assertEquals(STATUS_OK, matrixApi.dm_read(product, actual, actual.size.toLong()))

            val expected = doubleArrayOf(58.0, 64.0, 139.0, 154.0)
            assertArrayEquals(expected, actual, "Matrix multiplication result is incorrect")
        } finally {
            if (product != null) {
                matrixApi.dm_delete(product)
            }
            matrixApi.dm_delete(rhs)
            matrixApi.dm_delete(lhs)
        }
    }
}
