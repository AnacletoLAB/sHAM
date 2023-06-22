project(pybind11)
cmake_minimum_required(VERSION 3.0)
find_package(Git)

## STAMPA TUTTE LE VARIABILI DI AMBIENTE CREATE DA CMAKE
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

include(ExternalProject)

ExternalProject_Add(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
	#PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
	BUILD_IN_SOURCE 1
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    #CONFIGURE_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/pybind11-prefix/src/pybind11/cmake ."
	#INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}
    #BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

ExternalProject_Add_Step(
    pybind11 copyInclude
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/pybind11-prefix/src/pybind11/include/ ${CMAKE_CURRENT_SOURCE_DIR}
    #COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/pybind11-prefix/src/pybind11/include/pybind11/*.h ${CMAKE_CURRENT_SOURCE_DIR}
    #COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/detail
    #COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/pybind11-prefix/src/pybind11/include/pybind11/detail/*.h ${CMAKE_CURRENT_SOURCE_DIR}/detail
    DEPENDEES install
)

#set(zlib_LIB ${CMAKE_CURRENT_BINARY_DIR}/zlib-prefix/src/zlib/libz.a)
#set(pybind11_INCLUDEDIR ${CMAKE_CURRENT_BINARY_DIR}/pybind11/pybind11-prefix/src/pybind11/include/pybind11/$)
#set(pybind11_det_INCLUDEDIR ${CMAKE_CURRENT_BINARY_DIR}/pybind11/pybind11-prefix/src/pybind11/include/pybind11/detail/$)