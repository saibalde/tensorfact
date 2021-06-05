configure_file(Doxyfile.in Doxyfile @ONLY)

add_custom_target(docs ALL COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
                           WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/doc"
                           COMMENT "Generating documentation")
