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
#include "dense_matrix_api.h"
#include <algorithm>

size_t
safe_call(size_t (dm_double::*getter)() const noexcept, dm_ptr obj) noexcept {
    if (obj == nullptr) {
        return 0;
    }
    return (obj->matrix.*getter)();
}

extern "C" {
dm_ptr dm_new_empty() noexcept {
    try {
        auto* obj = new dm_storage {};
        return obj;
    } catch (const std::bad_alloc&) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

dm_ptr dm_new(size_t row_count, size_t col_count) noexcept {
    try {
        auto* obj = new dm_storage { dm_double(row_count, col_count) };
        return obj;
    } catch (const std::bad_alloc&) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

void dm_delete(dm_ptr obj) noexcept { delete obj; }

size_t dm_rows(dm_ptr obj) noexcept { return safe_call(&dm_double::rows, obj); }

size_t dm_cols(dm_ptr obj) noexcept { return safe_call(&dm_double::cols, obj); }

size_t dm_size(dm_ptr obj) noexcept { return safe_call(&dm_double::size, obj); }

dm_status dm_write(dm_ptr obj, const double* src, size_t value_count) noexcept {
    if (obj == nullptr) {
        return null;
    }
    if (src == nullptr && value_count != 0) {
        return null;
    }

    try {
        dm_double& matrix = obj->matrix;
        if (matrix.size() != value_count) {
            return bad_size;
        }

        if (value_count != 0) {
            std::copy_n(src, value_count, matrix.data());
        }
        return ok;
    } catch (const std::bad_alloc&) {
        return bad_alloc;
    } catch (const std::exception&) {
        return internal;
    } catch (...) {
        return internal;
    }
}

dm_status dm_read(dm_ptr obj, double* dst, size_t value_count) noexcept {
    if (obj == nullptr) {
        return null;
    }
    if (dst == nullptr && value_count != 0) {
        return null;
    }

    try {
        const dm_double& matrix = obj->matrix;
        if (matrix.size() != value_count) {
            return bad_size;
        }

        if (value_count != 0) {
            std::copy_n(matrix.data(), value_count, dst);
        }
        return ok;
    } catch (const std::bad_alloc&) {
        return bad_alloc;
    } catch (const std::exception&) {
        return internal;
    } catch (...) {
        return internal;
    }
}

dm_status dm_mul(dm_ptr lhs, dm_ptr rhs, dm_ptr* out_obj) noexcept {
    if (out_obj == nullptr) {
        return null;
    }
    *out_obj = nullptr;

    if (lhs == nullptr || rhs == nullptr) {
        return null;
    }

    try {
        const dm_double& left_matrix = lhs->matrix;
        const dm_double& right_matrix = rhs->matrix;

        if (left_matrix.cols() != right_matrix.rows()) {
            return bad_size;
        }

        dm_double product = dm_double::multiply(left_matrix, right_matrix);
        auto* obj = new dm_storage { std::move(product) };
        *out_obj = obj;
        return ok;
    } catch (const std::bad_alloc&) {
        return bad_alloc;
    } catch (const std::overflow_error&) {
        return bad_size;
    } catch (const std::invalid_argument&) {
        return bad_size;
    } catch (const std::exception&) {
        return internal;
    } catch (...) {
        return internal;
    }
}
}
