![matrix-centipede-Light](assets/lbanner.svg#gh-light-mode-only)![matrix-centipede-Dark](assets/dbanner.svg#gh-dark-mode-only)

[![version](https://img.shields.io/github/v/release/ninjaro/matrix-centipede?include_prereleases)](https://github.com/ninjaro/matrix-centipede/releases/latest)
[![Checks](https://github.com/ninjaro/matrix-centipede/actions/workflows/tests.yml/badge.svg)](https://github.com/ninjaro/matrix-centipede/actions/workflows/tests.yml)
[![Deploy](https://github.com/ninjaro/matrix-centipede/actions/workflows/html.yml/badge.svg)](https://github.com/ninjaro/matrix-centipede/actions/workflows/html.yml)
[![codecov](https://codecov.io/gh/ninjaro/matrix-centipede/graph/badge.svg?token=5XTA5DNO4S)](https://codecov.io/gh/ninjaro/matrix-centipede)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2288fe837def4b1b9d89f170e9d63594)](https://app.codacy.com/gh/ninjaro/matrix-centipede/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![license](https://img.shields.io/github/license/ninjaro/matrix-centipede?color=e6e6e6)](https://github.com/ninjaro/matrix-centipede/blob/master/license)

<p align="right">October 4th, 2025.</p>

> We are unfashioned creatures, but half made up, if one nerdier, fussier, and stricter than ourselves—such a benchmark
> ought to be—do not lend its aid to perfectionate our weak and faulty natures.

<p align="right">October 5th, 2025.</p>

> I agree with you; we are unfashioned creatures, if one younger, JetBrains-born, and island-named than ourselves—such a
> modern programming language ought to be—do play at baubles befitting its age, appropriate foreign artefacts, and
> desecrate the limbs of its forefathers.

## Requirements

* C++20 compiler (GCC or Clang recommended)
* CMake at least 3.28
* Optional:

    * **GTest**: for unit tests (`BUILD_TESTS=ON`)
    * **Google Benchmark**: for benchmarks (`BUILD_BENCHMARKS=ON`)
    * **Eigen3**: for extra test/bench comparisons if found
    * **Python 3 + matplotlib**: for plotting benchmark results via `scripts/bench_plot.py`
    * **llvm-cov + llvm-profdata** (Clang) or **gcovr** (GCC): for coverage reports (`ENABLE_COVERAGE=ON`)
* For the Java/Kotlin bindings:

  * **JDK 17** (configured via Gradle toolchains)
  * **Gradle 7.6+** (or an IDE with Gradle integration such as IntelliJ IDEA)

## Project Scripts

The `scripts/` directory contains utility helpers for common project workflows:

* `bench.slurm`: Submits a SLURM job that configures a fresh build tree on the
  cluster, enables the benchmark targets, and produces the benchmark plot with the
  Python virtual environment it bootstraps on the node.
* `bench_plot.py`: Parses Google Benchmark log files, groups results by
  algorithm and problem size, and renders a `GFLOPs` vs. `n` plot to an image using
  matplotlib.
* `brench.sh`: Convenience wrapper for running the same benchmark build and
  plot generation locally without SLURM by configuring the project in release mode
  and invoking the `bench_plot` target.
* `cov.sh`: Cleans previous coverage artifacts, reformats the C++ sources, and
  rebuilds the project with Clang in coverage mode before invoking the `cov/`
  target to export reports.
* `java_test.sh`: Builds the native matrix libraries required by the Java
  bindings and then runs the Gradle test suite with the correct native library
  path injected, validating the Java interface end-to-end.

## Documentation and Contributing

For detailed documentation, see the [Documentation](https://ninjaro.github.io/matrix-centipede/doc/) and for the latest
coverage report, see [Coverage](https://ninjaro.github.io/matrix-centipede/cov/).

## Security Policy

Please report any security issues using GitHub's private vulnerability reporting
or by emailing [yaroslav.riabtsev@rwth-aachen.de](mailto:yaroslav.riabtsev@rwth-aachen.de).
See the [security policy](.github/SECURITY.md) for full details.

## License

This project is open-source and available under the MIT License.
