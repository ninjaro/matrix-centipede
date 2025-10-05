cmake_minimum_required(VERSION 3.28)

project(matrix VERSION 1.0.0 LANGUAGES CXX)

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

add_library(matrix_lib
        src/dense_matrix.cpp
)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(matrix_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(matrix_lib PUBLIC include)

target_precompile_headers(matrix_lib PRIVATE
        include/dense_matrix.hpp
)

set_target_properties(matrix_lib PROPERTIES UNITY_BUILD ON)

add_executable(matrix src/main.cpp)

target_link_libraries(matrix PRIVATE matrix_lib)
if (OpenMP_CXX_FOUND)
    target_link_libraries(matrix PRIVATE OpenMP::OpenMP_CXX)
endif()

option(BUILD_TESTS "Build unit tests" ON)
option(COVERAGE "Enable coverage reporting" OFF)

if (BUILD_TESTS)
    find_package(GTest REQUIRED)

    if (UNIX)
        find_package(Threads REQUIRED)
    endif ()

    add_executable(unit_tests
            tests/main.cpp
            tests/dense_matrix_tests.cpp
    )

    target_link_libraries(unit_tests PRIVATE
            matrix_lib
            GTest::GTest
            GTest::Main
            Threads::Threads
    )
    if (OpenMP_CXX_FOUND)
        target_link_libraries(unit_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()

    enable_testing()
    add_test(NAME unit_tests COMMAND unit_tests)

    if (COVERAGE)
        message(STATUS "Coverage: ON, compiler is ${CMAKE_CXX_COMPILER_ID}")

        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(COV_FLAGS --coverage -O0)
            foreach (tgt IN ITEMS matrix_lib unit_tests)
                target_compile_options(${tgt} PUBLIC ${COV_FLAGS})
                target_link_libraries(${tgt} PUBLIC gcov)
            endforeach ()

            # custom target to run gcov/lcov
            find_program(LCOV lcov)
            find_program(GENHTML genhtml)
            add_custom_target(coverage
                    COMMENT "Generating GCC/gcov coverage report"
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} --build . --target unit_tests
                    COMMAND unit_tests
                    COMMAND ${LCOV} --capture --directory . --output-file coverage.info
                    COMMAND ${GENHTML} coverage.info --output-directory coverage_report
                    DEPENDS unit_tests
            )

        elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(COV_FLAGS
                    -fprofile-instr-generate
                    -fcoverage-mapping
                    -O0
            )
            set(PROFRAW_FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.profraw")
            set(PROFDATA_FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.profdata")
            set(COVERAGE_REPORT_DIR "${CMAKE_BINARY_DIR}/../cov")

            foreach (tgt IN ITEMS matrix_lib unit_tests)
                target_compile_options(${tgt} PUBLIC ${COV_FLAGS})
                target_link_options(${tgt} PUBLIC -fprofile-instr-generate)
            endforeach ()


            find_program(LLVM_PROFDATA llvm-profdata)
            find_program(LLVM_COV llvm-cov)
            add_custom_target(coverage
                    BYPRODUCTS ${PROFRAW_FILE} ${PROFDATA_FILE}
                    COMMENT "Generating Clang/LLVM-cov coverage report"
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} --build . --target unit_tests
                    COMMAND LLVM_PROFILE_FILE=${PROFRAW_FILE} $<TARGET_FILE:unit_tests>
                    COMMAND ${LLVM_PROFDATA} merge -sparse -output=${PROFDATA_FILE} ${PROFRAW_FILE}
                    COMMAND ${LLVM_COV} show $<TARGET_FILE:unit_tests>
                    -instr-profile=${PROFDATA_FILE}
                    -format=html
                    -output-dir=${COVERAGE_REPORT_DIR}
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
                    DEPENDS unit_tests matrix_lib
            )
        else ()
            message(WARNING "Coverage is only supported with GCC or Clang")
        endif ()
    endif ()

endif ()

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if (BUILD_TESTS AND ENABLE_ASAN)
    message(STATUS "AddressSanitizer ENABLED for unit_tests")
    target_compile_options(unit_tests PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
    target_link_options(unit_tests PRIVATE -fsanitize=address)
endif()