cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(TensorFact VERSION 0.3.0)

# dependencies

find_package(Armadillo REQUIRED)

find_package(Doxygen)

# main library

add_subdirectory(src)

# testing

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(TensorFact_ENABLE_TESTS "Build package tests" ON)
endif()

if(TensorFact_ENABLE_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(test)
endif()

# documentation

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(TensorFact_ENABLE_DOCS "Build package documentation" ON)
endif()

if(TensorFact_ENABLE_DOCS AND DOXYGEN_FOUND)
    add_subdirectory(doc)
endif()
