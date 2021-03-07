configure_file(Doxyfile.in Doxyfile @ONLY)

add_custom_target(docs ALL COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
                           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                           COMMENT "Generating documentation")
