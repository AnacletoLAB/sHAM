# megaDot for HAM / sHAM
# 2020 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
project(megaDot)
cmake_minimum_required(VERSION 3.12)

option(debug "Compile in debug mode" OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(OpenMP REQUIRED)

add_subdirectory(pybind11)

### C++ flags for GCC!
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mno-red-zone")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mstackrealign")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-asynchronous-unwind-tables")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")	# Questa dobbiamo tenercela
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")		# E pure questa

include_directories(.)
include_directories(${zlib_SOURCE_DIR})
include_directories(pybind11)
include_directories(${Python_INCLUDE_DIRS})

if(debug)
	message("____ DEBUG MODE ____")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
	message("____ RELEASE MODE ____")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
	# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")	# for Intel VTune
	# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")	# for Intel Advisor
endif()

SET(SOURCESMAIN	"megaDot_pybind11.cpp")

add_library(megaDot SHARED ${SOURCESMAIN})
set_property(TARGET megaDot PROPERTY POSITION_INDEPENDENT_CODE ON)
add_dependencies(megaDot pybind11)

