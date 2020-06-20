add_library(tt vector.cpp)
target_compile_features(tt PUBLIC cxx_std_11)
target_include_directories(tt PUBLIC ${PROJECT_SOURCE_DIR}/include
                                     ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(tt PUBLIC ${ARMADILLO_LIBRARIES})

add_library(tt::tt ALIAS tt)
