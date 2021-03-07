add_library(tensorfact svd.cpp
                       tt_tensor.cpp)
target_include_directories(tensorfact PUBLIC ${PROJECT_SOURCE_DIR}/include
                                             ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(tensorfact PUBLIC ${ARMADILLO_LIBRARIES}
                                        LAPACK::LAPACK
                                        BLAS::BLAS)
target_compile_features(tensorfact PUBLIC cxx_std_11)

add_library(TensorFact::tensorfact ALIAS tensorfact)
