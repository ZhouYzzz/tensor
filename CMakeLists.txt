# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("tensor")

# Include Local files
include_directories("include")

# Add CUDA support
find_package(cuda)
include_directories(${CUDA_INCLUDE_DIRS})

# Include sub-projects.
add_subdirectory ("src")

# UnitTest
add_subdirectory("test")
