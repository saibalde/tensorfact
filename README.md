# tensorfact

A C++ libary implementing various tensor factorizations.

## Dependencies

This library depends on the [BLAS++](https://bitbucket.org/icl/blaspp) and
[LAPACK++](https://bitbucket.org/icl/lapackpp) interfaces to BLAS and LAPACK
libraries for fast and accurate linear algebra operations.
[GoogleTest](https://github.com/google/googletest) is required only to build
the tests.

## Building

[CMake](https://cmake.org/) is used for compiling this library. For a basic
build, run the following commands from inside the repository root directory:
```sh
mkdir build
cd build
cmake \
    -D BLASPP_PREFIX_PATH=/blaspp/install/root \
    -D LAPACKPP_PREFIX_PATH=/lapackpp/install/root \
    -D GTEST_PREFIX_PATH=/googletest/install/root \
    ..
make
make test
```
