cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(TensorFact LANGUAGES CXX VERSION 0.1.0)

find_package(blaspp REQUIRED)
find_package(lapackpp REQUIRED)
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
