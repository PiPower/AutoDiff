cmake_minimum_required(VERSION 3.22.1)

project(LizardAutoDiff)

include_directories(/usr/local/cuda-12.3/include)
include_directories(/usr/include/)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	add_definitions(-DDEBUG)
endif()

add_subdirectory(./Utils)
add_subdirectory(./Tensor)
add_subdirectory(./Expressions)
add_subdirectory(./Graph)