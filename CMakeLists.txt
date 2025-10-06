cmake_minimum_required(VERSION 3.28)

project(matrix VERSION 1.0.0 LANGUAGES CXX)

option(BUILD_TESTS "Build unit tests (GTest)" ON)
option(BUILD_BENCHMARKS "Build benches if Google Benchmark is present" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-Og -g -fno-omit-frame-pointer
            -Werror
#            -Wno-deprecated-declarations

            -Wall
            -Wextra
            -Wpedantic

            -Wcast-align
            -Wcast-qual
            -Wconversion
            -Wctor-dtor-privacy
            -Wenum-compare
            -Wfloat-equal
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Woverloaded-virtual
            -Wredundant-decls
            -Wsign-conversion
            -Wsign-promo
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

find_package(Eigen3 3.3 QUIET)
add_library(dep_eigen INTERFACE)
if (Eigen3_FOUND)
    target_link_libraries(dep_eigen INTERFACE Eigen3::Eigen)
    target_compile_definitions(dep_eigen INTERFACE HAVE_EIGEN=1)
    message(STATUS "Eigen3 found: enabling Eigen in tests/benchmarks")
else()
    message(STATUS "Eigen3 not found: Eigen-based tests/benchmarks will be skipped")
endif()

if (BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if (GTest_FOUND)
        add_executable(unit_tests tests/dense_matrix_tests.cpp tests/dense_matrix_eigen_tests.cpp)
        target_link_libraries(unit_tests PRIVATE matrix dep_eigen GTest::gtest_main)
        add_test(NAME unit_tests COMMAND unit_tests)
    else ()
        message(STATUS "GTest not found. Skipping tests.")
    endif ()
endif ()

if (BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if (benchmark_FOUND)
        add_executable(bench_all benchmarks/benchmark_all.cpp)
        target_link_libraries(bench_all PRIVATE matrix dep_eigen benchmark::benchmark_main)
    else()
        message(STATUS "Google Benchmark not found. Skipping benches.")
    endif ()
endif ()