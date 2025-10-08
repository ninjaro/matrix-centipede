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
#ifndef MATRIX_CENTIPEDE_DENSE_MATRIX_API_H
#define MATRIX_CENTIPEDE_DENSE_MATRIX_API_H

#include "dense_matrix.hpp"

/**
 * @brief Export macro so the symbols remain visible to dynamic loaders.
 */
#define dm_api __attribute__((visibility("default")))

/**
 * @brief Alias for the double-precision dense matrix used by the C API.
 */
using dm_double = dm::dense_matrix<double>;

/**
 * @brief Status codes returned by the plain C interface.
 */
typedef enum { ok = 0, null, bad_size, bad_alloc, internal } dm_status;

/**
 * @brief Owning wrapper that keeps the matrix instance on the heap.
 */
struct dm_storage {
    dm_double matrix;
};

/**
 * @brief Convenience alias for pointers returned to API consumers.
 */
using dm_ptr = dm_storage*;

/**
 * @brief Helper that invokes a getter and normalises null receiver handling.
 *
 * The function guarantees a zero return for @c nullptr objects so that the JNI
 * layer can surface consistent behaviour without duplicating null checks.
 */
[[nodiscard]] size_t
safe_call(size_t (dm_double::*getter)() const noexcept, dm_ptr obj) noexcept;

extern "C" {

/**
 * @brief Allocates an empty storage object used as a sentinel.
 */
dm_ptr dm_new_empty(void) noexcept;
/**
 * @brief Allocates a matrix with the requested shape.
 */
dm_api dm_ptr dm_new(size_t row_count, size_t col_count) noexcept;
/**
 * @brief Destroys a storage object created by @ref dm_new or @ref dm_mul.
 */
dm_api void dm_delete(dm_ptr obj) noexcept;
/**
 * @brief Returns the number of rows in the referenced matrix, or zero for null.
 */
dm_api size_t dm_rows(dm_ptr obj) noexcept;
/**
 * @brief Returns the number of columns in the referenced matrix, or zero for
 * null.
 */
dm_api size_t dm_cols(dm_ptr obj) noexcept;
/**
 * @brief Returns the element count stored in the matrix, or zero for null.
 */
dm_api size_t dm_size(dm_ptr obj) noexcept;

/**
 * @brief Writes @p value_count entries from @p src into the matrix.
 */
dm_api dm_status
dm_write(dm_ptr obj, const double* src, size_t value_count) noexcept;
/**
 * @brief Copies matrix data into the provided destination buffer.
 */
dm_api dm_status dm_read(dm_ptr obj, double* dst, size_t value_count) noexcept;
/**
 * @brief Multiplies two matrices and stores the heap-allocated result in @p
 * out_obj.
 */
dm_api dm_status dm_mul(dm_ptr lhs, dm_ptr rhs, dm_ptr* out_obj) noexcept;

} // extern "C"

#endif // MATRIX_CENTIPEDE_DENSE_MATRIX_API_H
