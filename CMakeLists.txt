cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(TensorFact VERSION 0.2.0)

find_package(blaspp REQUIRED)
find_package(lapackpp REQUIRED)
find_package(HDF5 REQUIRED)
if(HDF5_IS_PARALLEL)
    find_package(MPI REQUIRED)
endif()

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
