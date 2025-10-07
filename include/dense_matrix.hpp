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

#ifndef MATRIX_CENTIPEDE_DENSE_MATRIX_HPP
#define MATRIX_CENTIPEDE_DENSE_MATRIX_HPP
#include <vector>

namespace dm {

enum class mul_algo { native, transpose, block_ijp, block_ipj };

template <class T>
concept matmul_scalar = std::default_initializable<T>
    && std::copy_constructible<T> && requires(T a, T b) {
           { a + b } -> std::same_as<T>;
           { a * b } -> std::same_as<T>;
           { a == b } -> std::convertible_to<bool>;
       };

template <matmul_scalar T = double> class dense_matrix {
public:
    dense_matrix() = default;
    dense_matrix(size_t rows, size_t cols);
    dense_matrix(size_t rows, size_t cols, const T* data);
    dense_matrix(size_t rows, size_t cols, std::initializer_list<T> init);
    dense_matrix(const dense_matrix&) = default;
    dense_matrix(dense_matrix&&) noexcept = default;
    dense_matrix& operator=(const dense_matrix&) = default;
    dense_matrix& operator=(dense_matrix&&) noexcept = default;
    ~dense_matrix() = default;

    [[nodiscard]] size_t rows() const noexcept;
    [[nodiscard]] size_t cols() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] T* data() noexcept;
    [[nodiscard]] const T* data() const noexcept;

    T& at(size_t r, size_t c);
    const T& at(size_t r, size_t c) const;
    T& operator()(size_t r, size_t c) noexcept;
    const T& operator()(size_t r, size_t c) const noexcept;

    [[nodiscard]] static dense_matrix multiply(
        const dense_matrix& a, const dense_matrix& b,
        mul_algo algo = mul_algo::native, size_t tile = 0
    );
    [[nodiscard]] dense_matrix mul(const dense_matrix& other) const;
    [[nodiscard]] dense_matrix operator*(const dense_matrix& other) const;

    [[nodiscard]] static dense_matrix
    add(const dense_matrix& a, const dense_matrix& b);
    [[nodiscard]] dense_matrix add(const dense_matrix& other) const;
    [[nodiscard]] dense_matrix operator+(const dense_matrix& other) const;
    [[nodiscard]] dense_matrix& operator+=(const dense_matrix& rhs);

    [[nodiscard]] bool operator==(const dense_matrix& rhs) const noexcept;

private:
    size_t row_count = 0;
    size_t col_count = 0;
    std::vector<T> values;

    [[nodiscard]] bool in_bounds(size_t r, size_t c) const noexcept;
    [[nodiscard]] size_t index_of(size_t r, size_t c) const noexcept;
    static size_t
    optimal_tile(size_t m = 0, size_t n = 0, size_t k = 0) noexcept;

    [[nodiscard]] dense_matrix transpose_tile(size_t tile = 0) const;
    [[nodiscard]] dense_matrix transpose() const;

    static dense_matrix
    mul_native(const dense_matrix& a, const dense_matrix& b);

    static dense_matrix
    mul_transpose(const dense_matrix& a, const dense_matrix& b, size_t tile);

    static dense_matrix
    mul_block_ipj(const dense_matrix& a, const dense_matrix& b, size_t tile);

    static dense_matrix
    mul_block_ijp(const dense_matrix& a, const dense_matrix& b, size_t tile);
};

}

#include "dense_matrix.tpp"

#endif // MATRIX_CENTIPEDE_DENSE_MATRIX_HPP
