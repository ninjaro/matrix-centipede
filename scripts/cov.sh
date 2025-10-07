#!/usr/bin/env bash

set -euo pipefail

cd ..

find . -regex '.*\.\(cpp\|hpp\|cc\|cxx\|tpp\)' -exec clang-format -style=file -i {} \;

clear
rm -rf build cov cov.log data/test*.dump

cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS=ON \
  -DCOVERAGE=ON

cmake --build build --target coverage --parallel $(nproc) | tee > cov.log