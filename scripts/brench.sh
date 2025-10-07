#!/usr/bin/env bash

set -euo pipefail

cd ..
rm -rf build bench/bench_log.txt bench/bench_plot.png
cmake -S . -B build   -DBUILD_BENCHMARKS=ON   -DDM_BENCH_MAX_N=1024   -DDM_BENCH_REPETITIONS=1   -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_plot