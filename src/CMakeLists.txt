# CMakeList.txt : CMake project for tensor, include source and define
# project specific logic here.
#

# Add source to this project's executable.
CUDA_ADD_LIBRARY (tensor STATIC "tensor.cpp"
                                "fft.cpp"
                                "cuda_helper.cpp"
                                "assign_add.cu")

CUDA_ADD_CUFFT_TO_TARGET(tensor)
CUDA_ADD_CUBLAS_TO_TARGET(tensor)
if(MSVC)
  link_directories("$ENV{CUDA_PATH}/lib/x64")
endif()
target_link_libraries(tensor glog cudnn)

add_executable (main "main.cpp")
target_link_libraries(main tensor)

# TODO: Add tests and install targets if needed.
