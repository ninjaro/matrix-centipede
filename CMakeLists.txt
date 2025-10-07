cmake_minimum_required(VERSION 3.28)

project(matrix VERSION 1.0.0 LANGUAGES CXX)

option(BUILD_TESTS "Build unit tests (GTest)" ON)
option(BUILD_BENCHMARKS "Build benches if Google Benchmark is present" ON)
option(COVERAGE "Enable code coverage instrumentation (tests only)" OFF)

if (COVERAGE AND NOT BUILD_TESTS)
    set(BUILD_TESTS ON CACHE BOOL "Build unit tests (forced by COVERAGE)" FORCE)
endif ()

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

add_library(matrix INTERFACE)
target_compile_features(matrix INTERFACE cxx_std_20)
target_include_directories(matrix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(matrix INTERFACE DM_HEADER_ONLY=1)
target_sources(matrix INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dense_matrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dense_matrix.tpp
)


add_library(matrix_api SHARED src/dense_matrix_api.cpp)
set_target_properties(matrix_api PROPERTIES
        OUTPUT_NAME matrix
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/native
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/native
)
target_include_directories(matrix_api
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(matrix_api PRIVATE matrix)
target_compile_definitions(matrix_api PRIVATE MC_DENSE_MATRIX_EXPORTS)

find_package(Eigen3 3.3 QUIET)
add_library(dep_eigen INTERFACE)
if (Eigen3_FOUND)
    target_link_libraries(dep_eigen INTERFACE Eigen3::Eigen)
    target_compile_definitions(dep_eigen INTERFACE HAVE_EIGEN=1)
endif ()

add_library(dep_coverage INTERFACE)
if (COVERAGE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(dep_coverage INTERFACE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(dep_coverage INTERFACE -fprofile-instr-generate -fcoverage-mapping)
        find_program(LLVM_COV llvm-cov REQUIRED)
        find_program(LLVM_PROFDATA llvm-profdata REQUIRED)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(dep_coverage INTERFACE --coverage -fno-inline -fno-inline-small-functions -fno-default-inline)
        target_link_options(dep_coverage INTERFACE --coverage)
        find_program(GCOVR gcovr REQUIRED)
    else ()
        message(FATAL_ERROR "Coverage only supported on GCC or Clang")
    endif ()
endif ()

if (BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if (GTest_FOUND)
        add_executable(unit_tests
                tests/dense_matrix_tests.cpp
                tests/dense_matrix_api_tests.cpp
                tests/dense_matrix_eigen_tests.cpp
        )
        target_link_libraries(unit_tests PRIVATE matrix matrix_api dep_eigen dep_coverage GTest::gtest_main)
        add_test(NAME unit_tests COMMAND unit_tests)
    endif ()
endif ()

if (BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if (benchmark_FOUND)
        add_executable(bench_all benchmarks/benchmark_all.cpp)
        target_link_libraries(bench_all PRIVATE matrix dep_eigen benchmark::benchmark_main)
    endif ()
endif ()

if (COVERAGE)
    set(COVERAGE_DIR "${CMAKE_BINARY_DIR}/../cov")

    if (TARGET unit_tests)
        add_custom_command(
                TARGET unit_tests POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory ${COVERAGE_DIR}
                VERBATIM
        )

        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(PROFRAW_FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.profraw")
            set(PROFDATA_FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.profdata")

            add_custom_target(coverage
                    COMMENT "Generating coverage with llvm-cov"
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} --build . --target unit_tests
                    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE=${PROFRAW_FILE} $<TARGET_FILE:unit_tests>
                    COMMAND ${LLVM_PROFDATA} merge -sparse -o ${PROFDATA_FILE} ${PROFRAW_FILE}
                    COMMAND ${LLVM_COV} show $<TARGET_FILE:unit_tests>
                    -instr-profile=${PROFDATA_FILE}
                    -format=html
                    -output-dir=${COVERAGE_DIR}
                    -ignore-filename-regex='.*/tests/.*'
                    -show-branches=count
                    -show-line-counts
                    -show-regions
                    -show-instantiations
                    -show-expansions
                    -use-color
                    -coverage-watermark=90,60
                    COMMAND ${LLVM_COV} export $<TARGET_FILE:unit_tests>
                    -instr-profile=${PROFDATA_FILE}
                    -format=lcov
                    -ignore-filename-regex='.*/tests/.*' > ${CMAKE_BINARY_DIR}/coverage.info
                    DEPENDS unit_tests
                    VERBATIM
            )

        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            add_custom_target(coverage
                    COMMENT "Generating coverage with gcovr"
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} --build . --target unit_tests
                    COMMAND unit_tests
                    COMMAND ${LCOV} --capture --directory . --output-file coverage.info
                    COMMAND ${GENHTML} coverage.info --output-directory coverage_report
                    DEPENDS unit_tests
            )
        endif ()

    else ()
        message(STATUS "COVERAGE=ON but tests are not built; 'coverage' target will be a no-op.")
    endif ()
endif ()


find_package(Python3 COMPONENTS Interpreter QUIET)

set(DM_BENCH_MAX_N 1024 CACHE STRING "Max n used by benchmark size generator")
set(DM_BENCH_REPETITIONS 1 CACHE STRING "Repetitions for Google Benchmark")

if (BUILD_BENCHMARKS AND benchmark_FOUND)
    target_compile_definitions(bench_all PRIVATE
            DM_BENCH_MAX_N=${DM_BENCH_MAX_N}
            DM_BENCH_REPETITIONS=${DM_BENCH_REPETITIONS}
    )
    string(TIMESTAMP BENCH_TS "%Y%m%d-%H%M%S" UTC)
    set(BENCH_DIR ${CMAKE_SOURCE_DIR}/bench)
    set(BENCH_LOG ${BENCH_DIR}/bench_log_${BENCH_TS}.txt)
    set(BENCH_PNG ${BENCH_DIR}/bench_plot_${BENCH_TS}.png)

    add_custom_command(
            OUTPUT ${BENCH_LOG}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${BENCH_DIR}
            COMMAND $<TARGET_FILE:bench_all>
            --benchmark_out=${BENCH_LOG}
            --benchmark_out_format=console
            DEPENDS bench_all
            COMMENT "Running Google Benchmark (check: ${BENCH_LOG})"
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
                COMMENT "Plotting GFLOPs vs n (check: ${BENCH_PNG})"
                VERBATIM
        )
        add_custom_target(bench_plot DEPENDS ${BENCH_PNG})
    else ()
        message(STATUS "Python3 not found, so 'bench_plot' target disabled (still have 'bench_run').")
    endif ()
endif ()
