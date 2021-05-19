add_library(tensorfact
    utils.cpp
    tt_tensor.cpp
    tt_svd.cpp
)
target_include_directories(tensorfact PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(tensorfact PRIVATE ${HDF5_C_INCLUDE_DIRS})
if(HDF5_IS_PARALLEL)
    target_link_libraries(tensorfact PRIVATE
        ${HDF5_C_LIBRARIES}
        MPI::MPI_CXX
        lapackpp
        blaspp
    )
else()
    target_link_libraries(tensorfact PRIVATE
        ${HDF5_C_LIBRARIES}
        lapackpp
        blaspp
    )
endif()
target_compile_definitions(tensorfact PUBLIC ${HDF5_C_DEFINITIONS})
target_compile_features(tensorfact PUBLIC cxx_std_11)
