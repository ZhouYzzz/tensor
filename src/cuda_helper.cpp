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
#include "cuda_helper.h"

class Cuda {
public:
  Cuda() {
    LOG(INFO) << "Setting up CUDA environment..";

    // check CUDA device then set
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    CHECK_GT(device_count, 0) << "No CUDA capable device found.";
    CHECK_CUDA(cudaSetDevice(0));
  }
  ~Cuda() {
    LOG(INFO) << "Tearing up CUDA environment..";

    // destroy handles if created
    if (cudnn_handle_)
      CHECK_CUDNN(cudnnDestroy(cudnn_handle_));
    if (cublas_handle_)
      CHECK_CUBLAS(cublasDestroy_v2(cublas_handle_));

    // reset CUDA device
    CHECK_CUDA(cudaDeviceReset());
  }
  inline cudnnHandle_t cudnn_handle() {
    if (!cudnn_handle_) {
      CHECK_CUDNN(cudnnCreate(&cudnn_handle_));
    }
    return cudnn_handle_;
  }
  inline cublasHandle_t cublas_handle() {
    if (!cublas_handle_) {
      CHECK_CUBLAS(cublasCreate_v2(&cublas_handle_));
    }
    return cublas_handle_;
  }
  inline cusolverDnHandle_t cusolverDn_handle() {
    if (!cusolverDn_handle_) {
      CHECK_CUSOLVER(cusolverDnCreate(&cusolverDn_handle_));
    }
    return cusolverDn_handle_;
  }
  inline float* workspace() { return workspace_; }
  inline void set_workspace(size_t size) {
    if (size > 0) {
      CHECK_LE(size, 2000000000) << "Workspace size exceeds";
      if (size > size_) {
        CHECK_CUDA(cudaFree(workspace_));
        CHECK_CUDA(cudaMalloc(&workspace_, size));
        size_ = size;
      }
    }
  }
private:
  cudnnHandle_t cudnn_handle_ = NULL;
  size_t size_ = 0;
  float* workspace_ = NULL;

  cublasHandle_t cublas_handle_ = NULL;

  cusolverDnHandle_t cusolverDn_handle_ = NULL;

};

// create global cuda_instance
static Cuda* cuda_instance_ = new Cuda();

cudnnHandle_t cudnn_handle() { return cuda_instance_->cudnn_handle(); }
cublasHandle_t cublas_handle() { return cuda_instance_->cublas_handle(); }
cusolverDnHandle_t cusolverDn_handle() { return cuda_instance_->cusolverDn_handle(); }
float* cudnn_get_workspace() { return cuda_instance_->workspace(); }
void cudnn_set_workspace(size_t size) { cuda_instance_->set_workspace(size); }
