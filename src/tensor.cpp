// tensor.cpp : Defines the entry point for the application.
//

#include "tensor.h"

#include <cstring> // memset

using namespace std;

SyncedMemory::SyncedMemory()
  :cpu_ptr_(NULL), gpu_ptr_(NULL), head_(UNINITIALIZED), size_(0) {
  CHECK_CUDA(cudaGetDevice(&device_));
}

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_)
    free(cpu_ptr_);
  if (gpu_ptr_)
    CHECK_CUDA(cudaFree(gpu_ptr_));
}

SyncedMemory::SyncedMemory(size_t size_in_bytes)
  :cpu_ptr_(NULL), gpu_ptr_(NULL), head_(UNINITIALIZED), size_(size_in_bytes){
  CHECK_CUDA(cudaGetDevice(&device_));
}

void SyncedMemory::checkDevice() {
#ifndef SINGLE_GPU
  int device;
  cudaGetDevice(&device);
  if (device != device_)
    FatalError("Device dismatch.");
  if (gpu_ptr_) {
    cudaPointerAttributes attributes;
    CHECK_CUDA(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    if (attributes.device != device_)
      FatalError("Device dismatch.");
  }
#endif
}

void SyncedMemory::to_cpu() {
  checkDevice();
  switch (head_) {
  case SyncedMemory::UNINITIALIZED:
    cpu_ptr_ = malloc(size_);
    memset(cpu_ptr_, 0, size_);
    head_ = HEAD_AT_CPU;
    break;
  case SyncedMemory::HEAD_AT_CPU:
    break;
  case SyncedMemory::HEAD_AT_GPU:
    if (cpu_ptr_ == NULL)
      cpu_ptr_ = malloc(size_);
    CHECK_CUDA(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDeviceToHost));
    head_ = SYNCED;
    break;
  case SyncedMemory::SYNCED:
    break;
  default:
    break;
  }
}

void SyncedMemory::to_gpu() {
  checkDevice();
  switch (head_) {
  case SyncedMemory::UNINITIALIZED:
    CHECK_CUDA(cudaMalloc(&gpu_ptr_, size_));
    CHECK_CUDA(cudaMemset(gpu_ptr_, 0, size_));
    head_ = HEAD_AT_GPU;
    break;
  case SyncedMemory::HEAD_AT_CPU:
    if (gpu_ptr_ == NULL)
      CHECK_CUDA(cudaMalloc(&gpu_ptr_, size_));
    CHECK_CUDA(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice));
    head_ = SYNCED;
    break;
  case SyncedMemory::HEAD_AT_GPU:
    break;
  case SyncedMemory::SYNCED:
    break;
  default:
    break;
  }
}

void* SyncedMemory::mutable_gpu_data() {
  to_gpu();
  return gpu_ptr_;
}
const void*   SyncedMemory::gpu_data() {
  to_gpu();
  return (const void*)gpu_ptr_;
}
void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  return cpu_ptr_;
}
const void*   SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}
