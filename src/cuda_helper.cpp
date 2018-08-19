#include "cuda_helper.h"

class Cuda {
public:
  Cuda() {
    LOG(INFO) << "Cuda instance";
    CHECK_CUDNN(cudnnCreate(&cudnn_handle_));
    CHECK_CUBLAS(cublasCreate_v2(&cublas_handle_));
  }
  ~Cuda() {
    LOG(INFO) << "Cuda destruction";
    CHECK_CUDNN(cudnnDestroy(cudnn_handle_));
    CHECK_CUBLAS(cublasDestroy_v2(cublas_handle_));
  }
  inline cudnnHandle_t cudnn_handle() { return cudnn_handle_; }
  inline cublasHandle_t cublas_handle() { return cublas_handle_; }
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
  cublasHandle_t cublas_handle_;
  size_t size_ = 0;
  float* workspace_ = NULL;
};

static Cuda* cuda_instance_ = new Cuda();

cudnnHandle_t cudnn_handle() { return cuda_instance_->cudnn_handle(); }
cublasHandle_t cublas_handle() { return cuda_instance_->cublas_handle(); }
float* cudnn_get_workspace() { return cuda_instance_->workspace(); }
void cudnn_set_workspace(size_t size) { cuda_instance_->set_workspace(size); }
