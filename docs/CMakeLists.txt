find_package(Doxygen REQUIRED dot)
find_package(Sphinx REQUIRED)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/conf.py
               @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index.rst
               ${CMAKE_CURRENT_BINARY_DIR}/index.rst
               COPYONLY)

add_custom_target(docs
                  ALL
                  ${SPHINX_EXECUTABLE} -b html ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/html
                  BYPRODUCTS ${CMAKE_CURRET_BINARY_DIR}/html/index.html
                  COMMENT "Generating documentation")
