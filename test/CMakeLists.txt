macro(TensorTrain_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} tt:tt GTest::gtest GTest::gtest_main)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()

TensorTrain_add_test(tt_vector_test tt_vector_test.cpp)
