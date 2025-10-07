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
#include "dense_matrix.hpp"
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#include <random>

using dm::dense_matrix;
using dm::mul_algo;

template <typename T>
using ematrix
    = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T> dense_matrix<T> eigen_to_dense(const ematrix<T>& e) {
    return dense_matrix<T>(
        static_cast<size_t>(e.rows()), static_cast<size_t>(e.cols()), e.data()
    );
}

template <typename T> ematrix<T> dense_to_eigen_copy(const dense_matrix<T>& d) {
    ematrix<T> e(
        static_cast<Eigen::Index>(d.rows()), static_cast<Eigen::Index>(d.cols())
    );
    std::memcpy(e.data(), d.data(), d.size() * sizeof(T));
    return e;
}

TEST(DenseMatrix_Eigen, Double_CompareWithEigen_isApprox) {
    const double tol = 1e-9;
    const std::vector<std::tuple<int, int, int>> shapes
        = { { 8, 8, 8 }, { 5, 37, 29 }, { 31, 7, 5 }, { 17, 13, 11 } };

    for (auto [m, k, n] : shapes) {
        ematrix<double> a = ematrix<double>::Random(m, k);
        ematrix<double> b = ematrix<double>::Random(k, n);
        ematrix<double> c = a * b;

        auto ad = eigen_to_dense(a);
        auto bd = eigen_to_dense(b);

        auto c_native
            = dense_matrix<double>::multiply(ad, bd, mul_algo::native);
        auto c_transp
            = dense_matrix<double>::multiply(ad, bd, mul_algo::transpose, 16);
        auto c_ijp
            = dense_matrix<double>::multiply(ad, bd, mul_algo::block_ijp, 16);
        auto c_ipj
            = dense_matrix<double>::multiply(ad, bd, mul_algo::block_ipj, 16);

        auto e_native = dense_to_eigen_copy(c_native);
        auto e_transp = dense_to_eigen_copy(c_transp);
        auto e_ijp = dense_to_eigen_copy(c_ijp);
        auto e_ipj = dense_to_eigen_copy(c_ipj);

        EXPECT_TRUE(c.isApprox(e_native, tol));
        EXPECT_TRUE(c.isApprox(e_transp, tol));
        EXPECT_TRUE(c.isApprox(e_ijp, tol));
        EXPECT_TRUE(c.isApprox(e_ipj, tol));
    }
}

TEST(DenseMatrix_Eigen, Int_ExactEquality_AgainstEigen) {
    constexpr int m = 6, k = 5, n = 7;

    ematrix<int> a(m, k), b(k, n);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-5, 5);
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        a.data()[i] = dist(rng);
    }
    for (Eigen::Index i = 0; i < b.size(); ++i) {
        b.data()[i] = dist(rng);
    }

    ematrix<int> c = a * b;

    auto ad = eigen_to_dense(a);
    auto bd = eigen_to_dense(b);
    auto c_ref = eigen_to_dense(c);

    auto c_native = dense_matrix<int>::multiply(ad, bd, mul_algo::native);
    auto c_transp = dense_matrix<int>::multiply(ad, bd, mul_algo::transpose, 8);
    auto c_ijp = dense_matrix<int>::multiply(ad, bd, mul_algo::block_ijp, 8);
    auto c_ipj = dense_matrix<int>::multiply(ad, bd, mul_algo::block_ipj, 8);

    EXPECT_TRUE(c_native == c_ref);
    EXPECT_TRUE(c_transp == c_ref);
    EXPECT_TRUE(c_ijp == c_ref);
    EXPECT_TRUE(c_ipj == c_ref);
}

#else

TEST(DenseMatrix_Eigen, SkippedWithoutEigen) {
    GTEST_SKIP() << "Eigen not available (HAVE_EIGEN not defined).";
}

#endif
