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
    if (cudnn_handle_inited_)
      CHECK_CUDNN(cudnnDestroy(cudnn_handle_));
    if (cublas_handle_inited_)
      CHECK_CUBLAS(cublasDestroy_v2(cublas_handle_));

    // reset CUDA device
    CHECK_CUDA(cudaDeviceReset());
  }
  inline cudnnHandle_t cudnn_handle() {
    if (!cudnn_handle_inited_) {
      CHECK_CUDNN(cudnnCreate(&cudnn_handle_));
      cudnn_handle_inited_ = true;
    }
    return cudnn_handle_;
  }
  inline cublasHandle_t cublas_handle() {
    if (!cublas_handle_inited_) {
      CHECK_CUBLAS(cublasCreate_v2(&cublas_handle_));
      cublas_handle_inited_ = true;
    }
    return cublas_handle_;
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
  cudnnHandle_t cudnn_handle_;
  bool cudnn_handle_inited_ = false;
  cublasHandle_t cublas_handle_;
  bool cublas_handle_inited_ = false;
  size_t size_ = 0;
  float* workspace_ = NULL;
};

static Cuda* cuda_instance_ = new Cuda();

cudnnHandle_t cudnn_handle() { return cuda_instance_->cudnn_handle(); }
cublasHandle_t cublas_handle() { return cuda_instance_->cublas_handle(); }
float* cudnn_get_workspace() { return cuda_instance_->workspace(); }
void cudnn_set_workspace(size_t size) { cuda_instance_->set_workspace(size); }
