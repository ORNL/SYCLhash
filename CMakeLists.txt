cmake_minimum_required(VERSION 3.8)

project(syclhash VERSION 1.0 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(rpath)
include(install)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(BUILD_TESTS "Build the tests accompanying this library." ON)
option(BUILD_DOCS "Build the documentation accompanying this library." ON)

## Setup SYCL
### Due to CMake limitations, hipSYCL requires C++ standard to be set manually
set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -std=c++17")
find_package(hipSYCL CONFIG REQUIRED)

##########  TARGETS  ################################################
add_library(syclhash INTERFACE)
FILE(GLOB SHASH_HEADERS include/syclhash/*.hpp)
#####################################################################

# configured header file stashing compile-time config options
file(MAKE_DIRECTORY include/syclhash)
configure_file(include/syclhash/config.hpp.in include/syclhash/config.hpp)
list(APPEND SHASH_HEADERS ${PROJECT_BINARY_DIR}/include/syclhash/config.hpp)

target_compile_features(syclhash INTERFACE cxx_std_17)
target_include_directories(syclhash INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                            $<INSTALL_INTERFACE:include>
                           )

# lib needs PIC when BUILD_SHARED_LIBS=ON
set_target_properties(syclhash PROPERTIES
                      POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
                      )

### Testing Instructions ###
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

### Installation Instructions ###
install_libs(TARGETS syclhash HEADERS ${SHASH_HEADERS})
