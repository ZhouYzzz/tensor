# CMakeList.txt : CMake project for tensor, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
CUDA_ADD_LIBRARY (tensor STATIC "tensor.cpp")

add_executable (main "main.cpp")
target_link_libraries(main tensor)

# TODO: Add tests and install targets if needed.
