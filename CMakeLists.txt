cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
project(tt LANGUAGES CXX VERSION 0.1.0)

find_package(Armadillo REQUIRED)

add_subdirectory(src)

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(tt_ENABLE_TESTS "Build the package tests" ON)
endif()

if(tt_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
