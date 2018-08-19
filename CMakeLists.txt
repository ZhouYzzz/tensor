# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.5)

project ("tensor")
# Enable CXX11 features
set (CMAKE_CXX_STANDARD 11)

# Include Local files
include_directories("include")

# Module search path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules")

# Add CUDA support
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
if(MSVC)
  link_directories("$ENV{CUDA_PATH}/lib/x64")
endif()

# Glog
if(MSVC)
  add_subdirectory("glog-0.3.5")
else()
  find_package(Glog)
endif()
include_directories(${Glog_INCLUDE_DIRS})

# Include sub-projects.
add_subdirectory ("src")

# UnitTest
add_subdirectory("test")