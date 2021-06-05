# tensorfact

A C++ libary implementing various tensor factorizations.

## Dependencies

This library depends on the following libraries:

*   [BLAS++](https://bitbucket.org/icl/blaspp)
*   [LAPACK++](https://bitbucket.org/icl/lapackpp)

[GoogleTest](https://github.com/google/googletest) is included as a submodule
with this library and is used for running tests.

## Building

[CMake](https://cmake.org/) is used for compiling this library. For a basic
build, run the following commands from inside the repository root directory:
```sh
mkdir build
cd build
cmake \
    -D blaspp_ROOT=/blaspp/install/prefix \
    -D lapackpp_ROOT=/lapackpp/install/prefix \
    ..
make
make test
```
