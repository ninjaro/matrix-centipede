cmake_minimum_required(VERSION 3.28)

project(matrix VERSION 1.0.0 LANGUAGES CXX)

option(BUILD_TESTS "Build unit tests (GTest)" ON)
option(BUILD_BENCHMARKS "Build benches if Google Benchmark is present" ON)
option(COVERAGE "Enable code coverage instrumentation (tests only)" OFF)

# If coverage is ON, force tests ON (header-only: coverage lives in test TUs)
if (COVERAGE AND NOT BUILD_TESTS)
    set(BUILD_TESTS ON CACHE BOOL "Build unit tests (forced by COVERAGE)" FORCE)
endif ()

set(COVERAGE_EXCLUDES
        "${CMAKE_SOURCE_DIR}/tests/.*"
        "${CMAKE_SOURCE_DIR}/benchmarks/.*"
        "${CMAKE_BINARY_DIR}/.*"
)
set(_COVERAGE_LLVM_EXCLUDE_ARGS "")
set(_COVERAGE_GCOVR_EXCLUDE_ARGS "")
foreach (_pattern IN LISTS COVERAGE_EXCLUDES)
    list(APPEND _COVERAGE_LLVM_EXCLUDE_ARGS "--ignore-filename-regex=${_pattern}")
    list(APPEND _COVERAGE_GCOVR_EXCLUDE_ARGS "--exclude=${_pattern}")
endforeach ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -Og -g -fno-omit-frame-pointer
            -Werror
            -Wall -Wextra -Wpedantic
            -Wcast-align -Wcast-qual -Wconversion
            -Wctor-dtor-privacy -Wenum-compare -Wfloat-equal
            -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual
            -Wredundant-decls -Wsign-conversion -Wsign-promo
    )
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s")
endif ()

# Header-only lib
add_library(matrix INTERFACE)
target_compile_features(matrix INTERFACE cxx_std_20)
target_include_directories(matrix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(matrix INTERFACE DM_HEADER_ONLY=1)
target_sources(matrix INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dense_matrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dense_matrix.tpp
)

# -------- Eigen (shared shim) -----------
find_package(Eigen3 3.3 QUIET)
add_library(dep_eigen INTERFACE)
if (Eigen3_FOUND)
    target_link_libraries(dep_eigen INTERFACE Eigen3::Eigen)
    target_compile_definitions(dep_eigen INTERFACE HAVE_EIGEN=1)
    message(STATUS "Eigen3 found: enabling Eigen in tests/benchmarks")
else ()
    message(STATUS "Eigen3 not found: Eigen-based tests/benchmarks will be skipped")
endif ()

# -------- Coverage (tests only) ----------
add_library(dep_coverage INTERFACE) # always exists; added only to test targets
if (COVERAGE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(dep_coverage INTERFACE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(dep_coverage INTERFACE -fprofile-instr-generate -fcoverage-mapping)
        find_program(LLVM_COV llvm-cov)
        find_program(LLVM_PROFDATA llvm-profdata)
        if (NOT LLVM_COV OR NOT LLVM_PROFDATA)
            message(FATAL_ERROR "COVERAGE=ON but llvm-cov/llvm-profdata not found")
        endif ()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(dep_coverage INTERFACE --coverage -fno-inline -fno-inline-small-functions -fno-default-inline)
        target_link_options(dep_coverage INTERFACE --coverage)
        find_program(GCOVR gcovr)
        if (NOT GCOVR)
            message(FATAL_ERROR "COVERAGE=ON but gcovr not found")
        endif ()
    else ()
        message(FATAL_ERROR "Coverage only supported on GCC or Clang")
    endif ()
endif ()

# -------- Tests --------------------------
if (BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if (GTest_FOUND)
        add_executable(unit_tests
                tests/dense_matrix_tests.cpp
                tests/dense_matrix_eigen_tests.cpp   # guarded with #ifdef HAVE_EIGEN
        )
        # IMPORTANT: add dep_coverage ONLY here (header-only code instantiates in this TU)
        target_link_libraries(unit_tests PRIVATE matrix dep_eigen dep_coverage GTest::gtest_main)
        add_test(NAME unit_tests COMMAND unit_tests)
    else ()
        message(STATUS "GTest not found. Skipping tests.")
    endif ()
endif ()

# -------- Benchmarks ---------------------
if (BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if (benchmark_FOUND)
        add_executable(bench_all benchmarks/benchmark_all.cpp)
        # No coverage for benches
        target_link_libraries(bench_all PRIVATE matrix dep_eigen benchmark::benchmark_main)
    else ()
        message(STATUS "Google Benchmark not found. Skipping benches.")
    endif ()
endif ()

# -------- coverage target (runs tests; emits HTML & lcov) ---
if (COVERAGE)
    set(COVERAGE_DIR "${CMAKE_BINARY_DIR}/../cov")

    if (TARGET unit_tests)
        # Common output locations
        add_custom_command(
                TARGET unit_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory ${COVERAGE_DIR}
                VERBATIM
        )

        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # llvm-cov flow (tools were already validated above)
            set(PROFRAW_FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.profraw")
            set(PROFDATA_FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.profdata")

            add_custom_target(coverage
                    COMMENT "Generating coverage with Clang/llvm-cov -> ${COVERAGE_DIR}"
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} --build . --target unit_tests
                    # Run tests with profiling
                    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE=${PROFRAW_FILE} $<TARGET_FILE:unit_tests>
                    # Merge and create reports
                    COMMAND ${LLVM_PROFDATA} merge -sparse -output=${PROFDATA_FILE} ${PROFRAW_FILE}
                    COMMAND ${LLVM_COV} show $<TARGET_FILE:unit_tests>
                    -instr-profile=${PROFDATA_FILE}
                    -format=html
                    -output-dir=${COVERAGE_DIR}
                    ${_COVERAGE_LLVM_EXCLUDE_ARGS}
                    -show-branches=count
                    -show-line-counts
                    -show-regions
                    -show-instantiations
                    -show-expansions
                    -Xdemangler=llvm-cxxfilt
                    -use-color
                    -coverage-watermark=90,60
                    COMMAND ${LLVM_COV} export $<TARGET_FILE:unit_tests>
                    -instr-profile=${PROFDATA_FILE}
                    -format=lcov
                    ${_COVERAGE_LLVM_EXCLUDE_ARGS}
                    > ${COVERAGE_DIR}/coverage.info
                    DEPENDS unit_tests
                    VERBATIM
            )

        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            # gcovr flow (gcovr was already validated above)
            add_custom_target(coverage
                    COMMENT "Generating coverage with GCC/gcovr -> ${COVERAGE_DIR}"
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} --build . --target unit_tests
                    COMMAND $<TARGET_FILE:unit_tests>
                    COMMAND ${GCOVR}
                    --root ${CMAKE_SOURCE_DIR}
                    --print-summary
                    ${_COVERAGE_GCOVR_EXCLUDE_ARGS}
                    --html --html-details --output ${COVERAGE_DIR}/index.html
                    COMMAND ${GCOVR}
                    --root ${CMAKE_SOURCE_DIR}
                    ${_COVERAGE_GCOVR_EXCLUDE_ARGS}
                    --lcov --output ${COVERAGE_DIR}/coverage.info
                    DEPENDS unit_tests
                    VERBATIM
            )
        else ()
            message(FATAL_ERROR "Coverage only supported on GCC or Clang")
        endif ()

    else ()
        message(STATUS "COVERAGE=ON but tests are not built; 'coverage' target will be a no-op.")
    endif ()
endif ()


# -------- Python for plotting (optional) -----------
find_package(Python3 COMPONENTS Interpreter QUIET)

# Let users tweak sizes/reps without editing sources
set(DM_BENCH_MAX_N 4096 CACHE STRING "Max n used by benchmark size generator")
set(DM_BENCH_REPETITIONS 1 CACHE STRING "Repetitions for Google Benchmark")

if (BUILD_BENCHMARKS AND benchmark_FOUND)
    # Pass macros to the bench target (your .cpp uses #ifndef defaults)
    target_compile_definitions(bench_all PRIVATE
            DM_BENCH_MAX_N=${DM_BENCH_MAX_N}
            DM_BENCH_REPETITIONS=${DM_BENCH_REPETITIONS}
    )

    set(BENCH_DIR ${CMAKE_SOURCE_DIR}/bench)
    set(BENCH_LOG ${BENCH_DIR}/bench_log.txt)
    set(BENCH_PNG ${BENCH_DIR}/bench_plot.png)

    add_custom_command(
            OUTPUT ${BENCH_LOG}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${BENCH_DIR}
            # Use GB's native output flag (no shell redirection needed)
            COMMAND $<TARGET_FILE:bench_all>
            --benchmark_out=${BENCH_LOG}
            --benchmark_out_format=console
            DEPENDS bench_all
            COMMENT "Running Google Benchmark -> ${BENCH_LOG}"
            VERBATIM
    )
    add_custom_target(bench_run DEPENDS ${BENCH_LOG})

    if (Python3_Interpreter_FOUND)
        add_custom_command(
                OUTPUT ${BENCH_PNG}
                COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/scripts/bench_plot.py
                ${BENCH_LOG} ${BENCH_PNG}
                DEPENDS ${BENCH_LOG}
                COMMENT "Plotting GFLOPs vs n -> ${BENCH_PNG}"
                VERBATIM
        )
        add_custom_target(bench_plot DEPENDS ${BENCH_PNG})
    else ()
        message(STATUS "Python3 not found -> 'bench_plot' target disabled (still have 'bench_run').")
    endif ()
endif ()
