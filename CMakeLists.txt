cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(TensorTrain LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
                      ${ARMADILLO_PREFIX_PATH}
                      ${GTEST_PREFIX_PATH})

find_package(Armadillo REQUIRED)
find_package(GTest)
find_package(Doxygen)

# create interface target

add_library(tt INTERFACE)
target_include_directories(tt INTERFACE ${PROJECT_SOURCE_DIR}/include
                                        ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(tt INTERFACE ${ARMADILLO_LIBRARIES})
target_compile_features(tt INTERFACE cxx_std_11)

add_library(tt::tt ALIAS tt)

# testing

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(TensorTrain_ENABLE_TESTS "Build package tests" ON)
endif()

if(TensorTrain_ENABLE_TESTS AND GTEST_FOUND)
    enable_testing()
    add_subdirectory(test)
endif()

# documentation

if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME)
    option(TensorTrain_ENABLE_DOCS "Build package documentation" ON)
endif()

if(TensorTrain_ENABLE_DOCS AND DOXYGEN_FOUND)
    add_subdirectory(doc)
endif()
