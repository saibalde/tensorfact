find_package(Doxygen REQUIRED dot)

if(DOXYGEN_FOUND)
    get_target_property(TensorTrain_PUBLIC_HEADER_DIR tt::tt INTERFACE_INCLUDE_DIRECTORIES)
    file(GLOB_RECURSE TensorTrain_PUBLIC_HEADERS ${TensorTrain_PUBLIC_HEADER_DIR}/*.hpp)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index.html ${CMAKE_CURRENT_BINARY_DIR}/index.html COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
                    DEPENDS ${TensorTrain_PUBLIC_HEADERS}
                    COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    MAIN_DEPENDENCY Doxyfile
                    COMMENT "Generating docs")

    add_custom_target(Doxygen ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/index.html)
else()
    message(WARNING "Doxygen not found - skipping documentation generation")
endif()
