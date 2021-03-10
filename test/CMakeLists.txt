macro(TensorFact_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_include_directories(${TESTNAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_link_libraries(${TESTNAME} PRIVATE TensorFact::tensorfact GTest::gtest GTest::gtest_main)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()

TensorFact_add_test(array_test array_test.cpp)
