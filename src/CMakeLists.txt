# CMakeList.txt : CMake project for tensor, include source and define
# project specific logic here.
#

# Add source to this project's executable.
CUDA_ADD_LIBRARY (tensor STATIC "tensor.cpp"
                                "tensor.cu"
                                "assign_add.cu")

CUDA_ADD_CUFFT_TO_TARGET(tensor)
target_link_libraries(tensor glog)

add_executable (main "main.cpp")
target_link_libraries(main tensor)

# TODO: Add tests and install targets if needed.
