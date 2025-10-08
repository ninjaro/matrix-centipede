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

/**
 * @brief Enumerates the available dense matrix multiplication algorithms.
 *
 * The different values select between naive, cache-aware, and block-based
 * implementations. They are exposed so that callers can benchmark or choose
 * the most appropriate strategy for their workload.
 */
enum class mul_algo { native, transpose, block_ijp, block_ipj };

/**
 * @brief Concept describing scalar types that can participate in matrix math.
 *
 * A type that satisfies @c matmul_scalar must be default constructible,
 * copyable, and support addition, multiplication, and equality comparison that
 * return objects convertible to @c bool. This keeps the matrix template
 * flexible while guaranteeing the operations required by the algorithms.
 */
template <class T>
concept matmul_scalar = std::default_initializable<T>
    && std::copy_constructible<T> && requires(T a, T b) {
           { a + b } -> std::same_as<T>;
           { a * b } -> std::same_as<T>;
           { a == b } -> std::convertible_to<bool>;
       };

/**
 * @brief Cache-friendly dense matrix implementation with multiple mul options.
 *
 * The class stores values in row-major order and provides construction,
 * element access, and arithmetic helpers. Multiplication can be dispatched to
 * several algorithms, ranging from the native triple loop to tiled variants
 * for better locality. The API is intentionally small yet expressive enough to
 * be consumed from both C++ and the C bindings declared elsewhere.
 *
 * @tparam T Scalar type that satisfies @ref matmul_scalar. @c double is used by
 * default for numerical stability.
 */
template <matmul_scalar T = double> class dense_matrix {
public:
    /**
     * @brief Constructs an empty 0x0 matrix.
     */
    dense_matrix() noexcept;
    /**
     * @brief Constructs a matrix with the given shape, value-initialising
     * cells.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    dense_matrix(size_t rows, size_t cols);
    /**
     * @brief Constructs a matrix by copying data from a contiguous buffer.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param data Pointer to at least @p rows*@p cols elements stored
     *             row-major.
     */
    dense_matrix(size_t rows, size_t cols, const T* data);
    /**
     * @brief Constructs a matrix from an initializer list.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param init Values in row-major order whose size must match rows*cols.
     */
    dense_matrix(size_t rows, size_t cols, std::initializer_list<T> init);
    /**
     * @brief Copy-constructs a matrix, duplicating all values.
     */
    dense_matrix(const dense_matrix&);
    /**
     * @brief Move-constructs a matrix, stealing ownership of the storage.
     */
    dense_matrix(dense_matrix&&) noexcept;
    /**
     * @brief Copy-assigns a matrix.
     */
    dense_matrix& operator=(const dense_matrix&);
    /**
     * @brief Move-assigns a matrix.
     */
    dense_matrix& operator=(dense_matrix&&) noexcept;
    /**
     * @brief Destroys the matrix and releases its storage.
     */
    ~dense_matrix();

    /**
     * @brief Returns the number of rows.
     */
    [[nodiscard]] size_t rows() const noexcept;
    /**
     * @brief Returns the number of columns.
     */
    [[nodiscard]] size_t cols() const noexcept;
    /**
     * @brief Returns the total number of stored elements.
     */
    [[nodiscard]] size_t size() const noexcept;
    /**
     * @brief Returns a pointer to the mutable contiguous storage.
     */
    [[nodiscard]] T* data() noexcept;
    /**
     * @brief Returns a pointer to the immutable contiguous storage.
     */
    [[nodiscard]] const T* data() const noexcept;

    /**
     * @brief Bounds-checked access to a matrix element.
     *
     * @param r Zero-based row index.
     * @param c Zero-based column index.
     * @throws std::out_of_range If the indices fall outside the matrix.
     */
    T& at(size_t r, size_t c);
    /**
     * @brief Const overload for bounds-checked access.
     *
     * @param r Zero-based row index.
     * @param c Zero-based column index.
     * @throws std::out_of_range If the indices fall outside the matrix.
     */
    const T& at(size_t r, size_t c) const;
    /**
     * @brief Unchecked mutable element accessor.
     *
     * @param r Zero-based row index (must be in range).
     * @param c Zero-based column index (must be in range).
     */
    T& operator()(size_t r, size_t c) noexcept;
    /**
     * @brief Unchecked const element accessor.
     *
     * @param r Zero-based row index (must be in range).
     * @param c Zero-based column index (must be in range).
     */
    const T& operator()(size_t r, size_t c) const noexcept;

    /**
     * @brief Multiplies two matrices using a selectable algorithm.
     *
     * @param a Left operand.
     * @param b Right operand.
     * @param algo Multiplication strategy. Defaults to @ref mul_algo::native.
     * @param tile Optional tile size used by tiled algorithms. When zero, a
     *             heuristic is chosen.
     * @returns The product matrix.
     * @throws std::invalid_argument If the shapes are incompatible or the
     *         algorithm value is invalid.
     */
    [[nodiscard]] static dense_matrix multiply(
        const dense_matrix& a, const dense_matrix& b,
        mul_algo algo = mul_algo::native, size_t tile = 0
    );
    /**
     * @brief Convenience wrapper that multiplies @c *this by @p other.
     *
     * Uses @ref mul_algo::block_ijp to balance cache friendliness and
     * performance.
     */
    [[nodiscard]] dense_matrix mul(const dense_matrix& other) const;
    /**
     * @brief Operator sugar for @ref mul().
     */
    [[nodiscard]] dense_matrix operator*(const dense_matrix& other) const;

    /**
     * @brief Adds two matrices of identical shape.
     *
     * @param a Left operand.
     * @param b Right operand.
     * @returns Matrix containing the pairwise sum.
     * @throws std::invalid_argument If the shapes differ.
     */
    [[nodiscard]] static dense_matrix
    add(const dense_matrix& a, const dense_matrix& b);
    /**
     * @brief Adds @p other to @c *this and returns the result as a new matrix.
     */
    [[nodiscard]] dense_matrix add(const dense_matrix& other) const;
    /**
     * @brief Operator sugar for @ref add(const dense_matrix&) const.
     */
    [[nodiscard]] dense_matrix operator+(const dense_matrix& other) const;
    /**
     * @brief Performs in-place addition of @p rhs.
     *
     * @throws std::invalid_argument If the shapes differ and neither matrix is
     *         empty.
     */
    [[nodiscard]] dense_matrix& operator+=(const dense_matrix& rhs);

    /**
     * @brief Compares matrices for shape and value equality.
     */
    [[nodiscard]] bool operator==(const dense_matrix& rhs) const noexcept;

private:
    /** Number of rows stored in the matrix. */
    size_t row_count = 0;
    /** Number of columns stored in the matrix. */
    size_t col_count = 0;
    /** Contiguous row-major storage for the matrix values. */
    std::vector<T> values;

    /**
     * @brief Checks whether the provided indices refer to a valid element.
     */
    [[nodiscard]] bool in_bounds(size_t r, size_t c) const noexcept;
    /**
     * @brief Converts a row/column pair to a flat index into @ref values.
     */
    [[nodiscard]] size_t index_of(size_t r, size_t c) const noexcept;
    /**
     * @brief Computes a cache-aware tile size, optionally clamped to problem
     * dimensions.
     */
    static size_t
    optimal_tile(size_t m = 0, size_t n = 0, size_t k = 0) noexcept;

    /**
     * @brief Produces a tiled transpose used by blocked algorithms.
     */
    [[nodiscard]] dense_matrix transpose_tile(size_t tile = 0) const;
    /**
     * @brief Computes the naive transpose of the matrix.
     */
    [[nodiscard]] dense_matrix transpose() const;

    /**
     * @brief Reference triple-loop matrix multiplication.
     */
    static dense_matrix
    mul_native(const dense_matrix& a, const dense_matrix& b);

    /**
     * @brief Multiplication that transposes @p b to improve spatial locality.
     */
    static dense_matrix
    mul_transpose(const dense_matrix& a, const dense_matrix& b, size_t tile);

    /**
     * @brief Blocked multiplication iterating in i-p-j order.
     */
    static dense_matrix
    mul_block_ipj(const dense_matrix& a, const dense_matrix& b, size_t tile);

    /**
     * @brief Blocked multiplication iterating in i-j-p order.
     */
    static dense_matrix
    mul_block_ijp(const dense_matrix& a, const dense_matrix& b, size_t tile);
};

}

#include "dense_matrix.tpp"

#endif // MATRIX_CENTIPEDE_DENSE_MATRIX_HPP
