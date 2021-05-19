# tensorfact

A C++ libary implementing various tensor factorizations.

## Dependencies

This library depends on the following libraries:

*   [BLAS++](https://bitbucket.org/icl/blaspp)
*   [LAPACK++](https://bitbucket.org/icl/lapackpp)
*   [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
*   MPI, if HDF5 was compiled with it

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
    -D HDF5_ROOT=/hdf5/install/prefix \
    -D MPI_ROOT=/mpi/install/prefix/ # only if HDF5 is compiled with MPI
    ..
make
make test
```
