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
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>
#include <stdexcept>

#include "dense_matrix.hpp"

namespace dm {
[[nodiscard]] constexpr size_t
safe_count(const size_t rows, const size_t cols) {
    if (rows != 0 && cols > (static_cast<size_t>(-1) / rows)) {
        throw std::overflow_error("rows*cols overflows size_t");
    }
    return rows * cols;
}

template <matmul_scalar T>
dense_matrix<T>::dense_matrix(const size_t rows, const size_t cols)
    : row_count(rows)
    , col_count(cols)
    , values(safe_count(rows, cols)) {
    assert(values.size() == rows * cols);
}

template <matmul_scalar T>
dense_matrix<T>::dense_matrix(
    const size_t rows, const size_t cols, const T* data
)
    : row_count(rows)
    , col_count(cols)
    , values(safe_count(rows, cols)) {
    const size_t n = values.size();
    if (n != 0 && data == nullptr) {
        throw std::invalid_argument("null data pointer for non-empty matrix");
    }
    if (n != 0) {
        std::ranges::copy(std::span(data, n), values.begin());
    }
    assert(values.size() == rows * cols);
}

template <matmul_scalar T>
dense_matrix<T>::dense_matrix(
    const size_t rows, const size_t cols, std::initializer_list<T> init
)
    : row_count(rows)
    , col_count(cols)
    , values(safe_count(rows, cols)) {
    const size_t n = values.size();
    if (init.size() != n) {
        throw std::invalid_argument("initializer_list size mismatch");
    }
    if (n != 0) {
        std::ranges::copy(init, values.begin());
    }
    assert(values.size() == rows * cols);
}

template <matmul_scalar T> size_t dense_matrix<T>::rows() const noexcept {
    return row_count;
}

template <matmul_scalar T> size_t dense_matrix<T>::cols() const noexcept {
    return col_count;
}

template <matmul_scalar T> size_t dense_matrix<T>::size() const noexcept {
    return values.size();
}

template <matmul_scalar T> T* dense_matrix<T>::data() noexcept {
    return values.data();
}

template <matmul_scalar T> const T* dense_matrix<T>::data() const noexcept {
    return values.data();
}

template <matmul_scalar T>
bool dense_matrix<T>::in_bounds(const size_t r, const size_t c) const noexcept {
    return r < row_count && c < col_count;
}

template <matmul_scalar T>
size_t
dense_matrix<T>::index_of(const size_t r, const size_t c) const noexcept {
    return r * col_count + c;
}

template <matmul_scalar T>
size_t dense_matrix<T>::optimal_tile(
    const size_t m, const size_t n, const size_t k
) noexcept {
    constexpr size_t l1_bytes = 32 * 1024;
    const double raw = std::sqrt(
        static_cast<double>(l1_bytes) / (3.0 * static_cast<double>(sizeof(T)))
    );
    auto tile = static_cast<size_t>(raw);

    const size_t vec = std::is_same_v<T, double> ? 8 : 16;
    if (tile < vec) {
        tile = vec;
    }
    tile = (tile / vec) * vec;

    constexpr size_t cap = 256;
    if (tile > cap) {
        tile = cap;
    }

    if (m) {
        tile = std::min(tile, m);
    }
    if (n) {
        tile = std::min(tile, n);
    }
    if (k) {
        tile = std::min(tile, k);
    }

    if (tile == 0) {
        tile = vec;
    }
    return tile;
}

template <matmul_scalar T>
T& dense_matrix<T>::at(const size_t r, const size_t c) {
    if (!in_bounds(r, c)) {
        throw std::out_of_range("dense_matrix::at: index out of range");
    }
    return values[index_of(r, c)];
}

template <matmul_scalar T>
const T& dense_matrix<T>::at(const size_t r, const size_t c) const {
    if (!in_bounds(r, c)) {
        throw std::out_of_range("dense_matrix::at const: index out of range");
    }
    return values[index_of(r, c)];
}

template <matmul_scalar T>
T& dense_matrix<T>::operator()(const size_t r, const size_t c) noexcept {
    assert(in_bounds(r, c));
    return values[index_of(r, c)];
}

template <matmul_scalar T>
const T&
dense_matrix<T>::operator()(const size_t r, const size_t c) const noexcept {
    assert(in_bounds(r, c));
    return values[index_of(r, c)];
}

template <matmul_scalar T>
dense_matrix<T>
dense_matrix<T>::add(const dense_matrix& a, const dense_matrix& b) {
    if (a.row_count != b.row_count || a.col_count != b.col_count) {
        throw std::invalid_argument("dense_matrix::add: shape mismatch");
    }
    dense_matrix out(a.row_count, a.col_count);
    std::ranges::transform(
        a.values, b.values, out.values.begin(), std::plus<> {}
    );
    return out;
}

template <matmul_scalar T>
dense_matrix<T> dense_matrix<T>::add(const dense_matrix& other) const {
    return add(*this, other);
}

template <matmul_scalar T>
dense_matrix<T> dense_matrix<T>::operator+(const dense_matrix& other) const {
    return add(*this, other);
}

template <matmul_scalar T>
bool dense_matrix<T>::operator==(const dense_matrix& rhs) const noexcept {
    return row_count == rhs.row_count && col_count == rhs.col_count
        && values == rhs.values;
}

template <matmul_scalar T>
dense_matrix<T> dense_matrix<T>::multiply(
    const dense_matrix& a, const dense_matrix& b, const mul_algo algo,
    const size_t tile
) {
    if (a.col_count != b.row_count) {
        throw std::invalid_argument(
            "dense_matrix::multiply: incompatible shapes"
        );
    }

    switch (algo) {
    case mul_algo::native:
        return mul_native(a, b);
    case mul_algo::transpose:
        return mul_transpose(a, b, tile);
    case mul_algo::block_ijp:
        return mul_block_ijp(a, b, tile);
    case mul_algo::block_ipj:
        return mul_block_ipj(a, b, tile);
    }
    throw std::invalid_argument("dense_matrix::multiply: unhandled mul_algo");
}

template <matmul_scalar T>
dense_matrix<T> dense_matrix<T>::mul(const dense_matrix& other) const {
    return multiply(*this, other, mul_algo::block_ijp);
}

template <matmul_scalar T>
dense_matrix<T> dense_matrix<T>::operator*(const dense_matrix& other) const {
    return mul(other);
}

template <matmul_scalar T>
dense_matrix<T> dense_matrix<T>::transpose_tile(size_t tile) const {
    const size_t m = row_count;
    const size_t n = col_count;
    dense_matrix t(n, m);

    if (m == 0 || n == 0) {
        return t;
    }
    if (m == 1 || n == 1) {
        std::ranges::copy(values, t.values.begin());
        return t;
    }

    if (!tile) {
        tile = optimal_tile(n, m);
    }

    const T* __restrict__ a = values.data();
    T* __restrict__ b = t.values.data();

    for (size_t j0 = 0; j0 < n; j0 += tile) {
        const size_t j1 = std::min(j0 + tile, n);
        for (size_t i0 = 0; i0 < m; i0 += tile) {
            const size_t i1 = std::min(i0 + tile, m);
            for (size_t j = j0; j < j1; ++j) {
                const size_t bj = j * m;
                const T* a_ptr = a + i0 * n + j;
                T* b_ptr = b + bj + i0;
                for (size_t i = i0; i < i1; ++i) {
                    *b_ptr++ = *a_ptr;
                    a_ptr += n;
                }
            }
        }
    }

    return t;
}

template <matmul_scalar T> dense_matrix<T> dense_matrix<T>::transpose() const {
    const size_t m = row_count;
    const size_t n = col_count;
    dense_matrix t(n, m);

    const T* __restrict__ a = values.data();
    T* __restrict__ b = t.values.data();

    for (size_t j = 0; j < n; ++j) {
        const size_t bj = j * m;
        const T* a_ptr = a + j;
        T* b_ptr = b + bj;
        for (size_t i = 0; i < m; ++i) {
            *b_ptr++ = *a_ptr;
            a_ptr += n;
        }
    }
    return t;
}

template <matmul_scalar T>
dense_matrix<T>
dense_matrix<T>::mul_native(const dense_matrix& a, const dense_matrix& b) {
    const size_t m = a.row_count;
    const size_t k = a.col_count;
    assert(k == b.row_count);
    const size_t n = b.col_count;

    dense_matrix out(m, n);
    if (out.size() == 0) {
        return out;
    }

    const T* a_ptr = a.values.data();
    const T* b_ptr = b.values.data();
    T* __restrict__ c_ptr = out.values.data();

    for (size_t i = 0; i < m; ++i) {
        const size_t a_off = i * k;
        T* __restrict__ c_row = c_ptr + i * n;

        for (size_t p = 0; p < k; ++p) {
            const T a_ip = a_ptr[a_off + p];
            const T* b_row = b_ptr + p * n;

            T* __restrict__ c_acc = c_row;
            const T* b_acc = b_row;
            for (size_t j = 0; j < n; ++j) {
                *c_acc++ += a_ip * *b_acc++;
            }
        }
    }
    return out;
}

template <matmul_scalar T>
dense_matrix<T> dense_matrix<T>::mul_transpose(
    const dense_matrix& a, const dense_matrix& b, size_t tile
) {
    const size_t m = a.row_count;
    const size_t k = a.col_count;
    assert(k == b.row_count);
    const size_t n = b.col_count;

    dense_matrix out(m, n);
    if (out.size() == 0) {
        return out;
    }

    dense_matrix bt = tile ? b.transpose_tile(tile) : b.transpose();

    const T* __restrict__ a_ptr = a.values.data();
    const T* __restrict__ bt_ptr = bt.values.data();
    T* __restrict__ c_ptr = out.values.data();

    for (size_t i = 0; i < m; ++i) {
        const T* __restrict__ a_row = a_ptr + i * k;
        T* __restrict__ c_row = c_ptr + i * n;
        for (size_t j = 0; j < n; ++j) {
            const T* __restrict__ bt_row = bt_ptr + j * k;
            T sum {};
            for (size_t p = 0; p < k; ++p) {
                sum += a_row[p] * bt_row[p];
            }
            c_row[j] = sum;
        }
    }

    return out;
}

template <matmul_scalar T>
dense_matrix<T> dense_matrix<T>::mul_block_ipj(
    const dense_matrix& a, const dense_matrix& b, size_t tile
) {
    const size_t m = a.row_count;
    const size_t k = a.col_count;
    assert(k == b.row_count);
    const size_t n = b.col_count;

    if (tile == 0) {
        tile = optimal_tile(n, m, k);
    }

    dense_matrix out(m, n);
    if (out.size() == 0) {
        return out;
    }

    const T* a_ptr = a.values.data();
    const T* b_ptr = b.values.data();
    T* __restrict__ c_ptr = out.values.data();

    for (size_t i0 = 0; i0 < m; i0 += tile) {
        const size_t i1 = std::min(i0 + tile, m);
        for (size_t p0 = 0; p0 < k; p0 += tile) {
            const size_t p1 = std::min(p0 + tile, k);
            for (size_t j0 = 0; j0 < n; j0 += tile) {
                const size_t j1 = std::min(j0 + tile, n);
                const size_t w = j1 - j0;

                for (size_t i = i0; i < i1; ++i) {
                    const size_t ai = i * k;
                    T* c_tile = c_ptr + i * n + j0;

                    for (size_t p = p0; p < p1; ++p) {
                        const T a_ip = a_ptr[ai + p];
                        const T* b_tile = b_ptr + p * n + j0;

                        T* c_acc = c_tile;
                        const T* b_acc = b_tile;
                        for (size_t off = 0; off < w; ++off) {
                            *c_acc++ += a_ip * *b_acc++;
                        }
                    }
                }
            }
        }
    }
    return out;
}

template <matmul_scalar T>
dense_matrix<T> dense_matrix<T>::mul_block_ijp(
    const dense_matrix& a, const dense_matrix& b, size_t tile
) {
    const size_t m = a.row_count;
    const size_t k = a.col_count;
    assert(k == b.row_count);
    const size_t n = b.col_count;

    if (tile == 0) {
        tile = optimal_tile(n, m, k);
    }

    dense_matrix out(m, n);
    if (out.size() == 0) {
        return out;
    }

    const T* a_ptr = a.values.data();
    const T* b_ptr = b.values.data();
    T* __restrict__ c_ptr = out.values.data();

    for (size_t i0 = 0; i0 < m; i0 += tile) {
        const size_t i1 = std::min(i0 + tile, m);
        for (size_t j0 = 0; j0 < n; j0 += tile) {
            const size_t j1 = std::min(j0 + tile, n);
            for (size_t p0 = 0; p0 < k; p0 += tile) {
                const size_t p1 = std::min(p0 + tile, k);

                for (size_t i = i0; i < i1; ++i) {
                    const size_t ai = i * k;
                    const size_t ci = i * n;

                    for (size_t p = p0; p < p1; ++p) {
                        const T a_ip = a_ptr[ai + p];
                        const size_t bp = p * n;

                        T* c_tile = c_ptr + ci + j0;
                        const T* b_tile = b_ptr + bp + j0;

                        for (size_t j = j0; j < j1; ++j) {
                            *c_tile++ += a_ip * *b_tile++;
                        }
                    }
                }
            }
        }
    }
    return out;
}
}
