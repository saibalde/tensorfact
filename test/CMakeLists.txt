add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")

mark_as_advanced(
    BUILD_GTEST
    BUILD_GMOCK
    BUILD_SHARED_LIBS
    gtest_build_samples
    gtest_build_tests
    gtest_disable_pthreads
    gtest_force_shared_crt
    gtest_hide_internal_symbols
    gmock_build_tests
)

set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)

macro(TensorFact_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_include_directories(${TESTNAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${ARMADILLO_INCLUDE_DIRS}
    )
    target_link_libraries(${TESTNAME} PRIVATE
        ${ARMADILLO_LIBRARIES}
        gtest
        tensorfact
    )
    gtest_discover_tests(${TESTNAME}
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()

TensorFact_add_test(thin_lq_test thin_lq_test.cpp)
TensorFact_add_test(truncated_svd_test truncated_svd_test.cpp)
TensorFact_add_test(tt_tensor_test tt_tensor_test.cpp)
TensorFact_add_test(tt_svd_test tt_svd_test.cpp)
