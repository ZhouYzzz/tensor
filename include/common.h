#pragma once

#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudnn.h>
#include <cufft.h>

#include <glog/logging.h>

// (default) SINGLE GPU
#define SINGLE_GPU

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

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CUDA_NUM_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
