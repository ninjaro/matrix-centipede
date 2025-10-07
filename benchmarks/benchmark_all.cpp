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

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#endif

using dm::dense_matrix;
using dm::mul_algo;

#ifndef DM_BENCH_REPETITIONS
#define DM_BENCH_REPETITIONS 1
#endif

#ifndef DM_BENCH_MAX_N
#define DM_BENCH_MAX_N 1536
#endif

static void sizes(benchmark::internal::Benchmark* b) {
    for (const int n : { 32, 48, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768,
                         1024, 1536, 2048, 3072, 4096, 6144, 8192 }) {
        if (n > DM_BENCH_MAX_N) {
            break;
        }
        b->Arg(n);
    }
}

static constexpr std::size_t pattern_modulus = 257;
static constexpr int pattern_offset = 128;

static double pattern_value_from_index(const std::size_t idx) {
    const auto v
        = static_cast<std::ptrdiff_t>(idx % pattern_modulus) - pattern_offset;
    return static_cast<double>(v);
}

static const std::vector<double>& pattern_buffer() {
    static std::vector<double> buf([] {
        constexpr std::size_t n = DM_BENCH_MAX_N;
        std::vector<double> tmp;
        tmp.resize(n * n);
        for (std::size_t r = 0; r < n; ++r) {
            const std::size_t base = r * n;
            for (std::size_t c = 0; c < n; ++c) {
                const std::size_t idx = base + c;
                tmp[idx] = pattern_value_from_index(idx);
            }
        }
        return tmp;
    }());
    return buf;
}

static void fill_dm_from_pattern(dense_matrix<double>& m) {
    const auto& src = pattern_buffer();
    constexpr std::size_t pitch = DM_BENCH_MAX_N;
    const std::size_t n = m.cols();
    double* dst = m.data();
    for (std::size_t r = 0; r < n; ++r) {
        const double* row_src = &src[r * pitch];
        std::memcpy(dst + r * n, row_src, n * sizeof(double));
    }
}

#ifdef HAVE_EIGEN
using ematrix
    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static void fill_eigen_from_pattern(ematrix& m) {
    const auto& src = pattern_buffer();
    constexpr std::size_t pitch = DM_BENCH_MAX_N;
    const std::size_t n = static_cast<std::size_t>(m.cols());
    double* dst = m.data();
    for (std::size_t r = 0; r < static_cast<std::size_t>(m.rows()); ++r) {
        const double* row_src = &src[r * pitch];
        std::memcpy(dst + r * n, row_src, n * sizeof(double));
    }
}
#endif

static void report_flops(benchmark::State& state, const std::size_t n) {
    const double it = static_cast<double>(state.iterations());
    const double ops = it * (2.0 * static_cast<double>((n - 1) * n * n));
    state.counters["FLOPs"]
        = benchmark::Counter(ops, benchmark::Counter::kIsRate);
}

static void BM_DM(benchmark::State& state, const mul_algo algo) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    dense_matrix<double> a(n, n);
    dense_matrix<double> b(n, n);
    fill_dm_from_pattern(a);
    fill_dm_from_pattern(b);

    for (auto _ : state) {
        benchmark::DoNotOptimize(a.data());
        benchmark::DoNotOptimize(b.data());
        auto c = dense_matrix<double>::multiply(a, b, algo);
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }
    report_flops(state, n);
}

#ifdef HAVE_EIGEN
static void BM_Eigen(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    ematrix a(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    ematrix b(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    fill_eigen_from_pattern(a);
    fill_eigen_from_pattern(b);

    for (auto _ : state) {
        benchmark::DoNotOptimize(a.data());
        benchmark::DoNotOptimize(b.data());
        ematrix c = a * b;
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }
    report_flops(state, n);
}
#endif

BENCHMARK_CAPTURE(BM_DM, native, mul_algo::native)
    ->ArgName("n")
    ->Apply(sizes)
    ->Repetitions(DM_BENCH_REPETITIONS)
    ->ReportAggregatesOnly(true);

BENCHMARK_CAPTURE(BM_DM, transpose, mul_algo::transpose)
    ->ArgName("n")
    ->Apply(sizes)
    ->Repetitions(DM_BENCH_REPETITIONS)
    ->ReportAggregatesOnly(true);

BENCHMARK_CAPTURE(BM_DM, block_ijp, mul_algo::block_ijp)
    ->ArgName("n")
    ->Apply(sizes)
    ->Repetitions(DM_BENCH_REPETITIONS)
    ->ReportAggregatesOnly(true);

BENCHMARK_CAPTURE(BM_DM, block_ipj, mul_algo::block_ipj)
    ->ArgName("n")
    ->Apply(sizes)
    ->Repetitions(DM_BENCH_REPETITIONS)
    ->ReportAggregatesOnly(true);

// #ifdef HAVE_EIGEN
// BENCHMARK(BM_Eigen)
//     ->ArgName("n")
//     ->Apply(sizes)
//     ->Repetitions(DM_BENCH_REPETITIONS)
//     ->ReportAggregatesOnly(true);
// #endif

BENCHMARK_MAIN();
