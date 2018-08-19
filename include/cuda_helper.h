/*The MIT License

Copyright (c) 2018 Yizhuang ZHOU

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// CUDA_HELPER.H defines frequently used macros, supports CUDA version 9.2
#pragma once

#include <sstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudnn.h>
#include <cufft.h>
#include <cublas.h>

#include <glog/logging.h>

#define FatalError(s) do {                                           \
  std::stringstream _where, _message;                                \
  _where << __FILE__ << ':' << __LINE__;                             \
  _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
  std::cerr << _message.str() << "\nAborting...\n";                  \
  cudaDeviceReset();                                                 \
  exit(1);                                                           \
} while(0)

#define CHECK_CUDNN(status) do {                                     \
  std::stringstream _error;                                          \
  if (status != CUDNN_STATUS_SUCCESS) {                              \
    _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
    FatalError(_error.str());                                        \
  }                                                                  \
} while(0)

#define CHECK_CUDA(status) do {                                      \
  std::stringstream _error;                                          \
  if (status != 0) {                                                 \
    _error << "Cuda failure: " << cudaGetErrorString(status);        \
    FatalError(_error.str());                                        \
  }                                                                  \
} while(0)

#define CHECK_CUFFT(status) do {                                     \
  std::stringstream _error;                                          \
  if (status != CUFFT_SUCCESS) {                                     \
    _error << "Cufft failure: ";                                     \
    FatalError(_error.str());                                        \
  }                                                                  \
} while(0)

#define CHECK_CUBLAS(status) do {                                    \
  std::stringstream _error;                                          \
  if (status != CUBLAS_STATUS_SUCCESS) {                             \
    _error << "CuBLAS failure: ";                                    \
    FatalError(_error.str());                                        \
  }                                                                  \
} while(0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                       \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;                \
       i < (n);                                                      \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CUDA_NUM_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

cudnnHandle_t cudnn_handle();
cublasHandle_t cublas_handle();
float* cudnn_get_workspace();
void cudnn_set_workspace(size_t size);
