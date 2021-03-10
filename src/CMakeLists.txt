add_library(tensorfact TensorFact_Array.cpp)
target_include_directories(tensorfact PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(tensorfact PRIVATE lapackpp blaspp)
target_compile_features(tensorfact PUBLIC cxx_std_11)
