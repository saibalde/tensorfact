cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(TensorFact LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
                      ${ARMADILLO_PREFIX_PATH}
                      ${GTEST_PREFIX_PATH})

find_package(Armadillo REQUIRED)
find_package(GTest)
find_package(Doxygen)

# main library

add_subdirectory(src)

# testing

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(TensorFact_ENABLE_TESTS "Build package tests" ON)
endif()

if(TensorFact_ENABLE_TESTS AND GTest_FOUND)
    enable_testing()
    add_subdirectory(test)
endif()

# documentation

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(TensorFact_ENABLE_DOCS "Build package documentation" ON)
endif()

if(TensorFact_ENABLE_DOCS AND DOXYGEN_FOUND)
    add_subdirectory(doc)
endif()
