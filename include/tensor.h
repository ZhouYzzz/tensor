// tensor.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <memory>

#include <cuda.h>
#include <cudnn.h>

#include "common.h"

/* Synced Memory between CPU and GPU */
class SyncedMemory {
public:
  SyncedMemory();
  ~SyncedMemory();
  explicit SyncedMemory(size_t size_in_bytes);

  void to_cpu();
  void to_gpu();

  void* mutable_gpu_data();
  const void*   gpu_data();
  void* mutable_cpu_data();
  const void*   cpu_data();

  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }
private:
  void checkDevice();
  void* cpu_ptr_;
  void* gpu_ptr_;
  SyncedHead head_;
  size_t size_;
  int device_;
};

/* Tensor class */
template <typename T>
class Tensor {
public:
  Tensor() {}
  explicit Tensor(unsigned n, unsigned c, unsigned h, unsigned w) :
  n_(n), c_(c), h_(h), w_(w), count_(n*c*h*w), size_(n*c*h*w*sizeof(T)) {
    mem_.reset(new SyncedMemory(size_));
  }
  inline unsigned n() { return n_; }
  inline unsigned c() { return c_; }
  inline unsigned h() { return h_; }
  inline unsigned w() { return w_; }
  inline size_t size() { return size_; }
  inline size_t count() { return count_; }

  // T* mutable_gpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) {
  //   return reinterpret_cast<T*>(mem_->mutable_gpu_data()) + offset(n, c, h, w);
  // }
  // const T*   gpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) {
  //   return reinterpret_cast<const T*>(mem_->gpu_data()) + offset(n, c, h, w);
  // }
  // T* mutable_cpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) {
  //   return reinterpret_cast<T*>(mem_->mutable_cpu_data()) + offset(n, c, h, w);
  // }
  // const T*   cpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) {
  //   return reinterpret_cast<const T*>(mem_->cpu_data()) + offset(n, c, h, w);
  // }

  // template func
  template <typename DT = T>
  DT* mutable_gpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) {
    CHECK_EQ(sizeof(T), sizeof(DT));
    return reinterpret_cast<DT*>(mem_->mutable_gpu_data()) + offset(n, c, h, w);
  }
  template <typename DT = T>
  const DT*   gpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) {
    CHECK_EQ(sizeof(T), sizeof(DT));
    return reinterpret_cast<const DT*>(mem_->gpu_data()) + offset(n, c, h, w);
  }
  template <typename DT = T>
  DT* mutable_cpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) {
    CHECK_EQ(sizeof(T), sizeof(DT));
    return reinterpret_cast<DT*>(mem_->mutable_cpu_data()) + offset(n, c, h, w);
  }
  template <typename DT = T>
  const DT*   cpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) {
    CHECK_EQ(sizeof(T), sizeof(DT));
    return reinterpret_cast<const DT*>(mem_->cpu_data()) + offset(n, c, h, w);
  }
private:
  inline int offset(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) {
    return w + w_ * (h + h_ * (c + c_ * n));
  }
  
  std::shared_ptr<SyncedMemory> mem_;

  unsigned n_, c_, h_, w_;
  size_t size_, count_;

  //DISABLE_ASSIGN_COPY(Tensor);
};

template <typename T>
std::ostream &operator<<(std::ostream &os, Tensor<T>& t) {
  os << "Tensor" << "<"
    << t.n() << ","
    << t.c() << ","
    << t.h() << ","
    << t.w() << ","
    << ">" << std::endl;
  for (int n = 0; n < t.n(); n++) {
  if (t.h() <= 11 && t.w() <= 11) {
    for (int i = 0; i < t.h(); i++) {
      os << "  ";
      for (int j = 0; j < t.w(); j++)
        os << *(t.cpu_data(n, 0, i, j)) << ",";
      os << std::endl;
    }
    os << std::endl;
  }
  }
  return os;
}

// TODO: Reference additional headers your program requires here.
