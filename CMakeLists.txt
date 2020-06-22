cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
project(TensorTrain LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(Armadillo REQUIRED)

add_subdirectory(src)

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(TensorTrain_ENABLE_TESTS "Build package tests" ON)
endif()

if(TensorTrain_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(TensorTrain_ENABLE_DOCS "Build package documentation" ON)
endif()

if(TensorTrain_ENABLE_DOCS)
    add_subdirectory(docs)
endif()
