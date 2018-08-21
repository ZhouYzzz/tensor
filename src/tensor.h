// tensor.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <sstream>
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

  // Since SyncedMemory is managed by shared_ptr, copy and assignment
  // are not permitted.
  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};

/* Tensor class */
template <typename T>
class Tensor {
public:
  Tensor() {}
  explicit Tensor(unsigned n, unsigned c, unsigned h, unsigned w);
  // perform shallow/deep copy
  Tensor(const Tensor& copy, bool deep_copy);
  
  // WIP, TODO: refer to cv::Mat::create() [core\src\matrix.cpp]
  // create a new memory block for access (usually as an output arg)
  void create(unsigned n, unsigned c, unsigned h, unsigned w);

  // WIP: reshape only, without changing the underlying data, but the shape information
  void reshape(unsigned n, unsigned c, unsigned h, unsigned w);

  Tensor<T>& reshape(unsigned n, unsigned c, unsigned h, unsigned w) const;

  Tensor<T> deep_copy() const;

  std::string property_string();

  // We use the default copy and assignment constructors here
  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor&) = default;

  inline unsigned n() const { return n_; }
  inline unsigned c() const { return c_; }
  inline unsigned h() const { return h_; }
  inline unsigned w() const { return w_; }
  inline size_t size() const { return size_; }
  inline size_t count() const { return count_; }

  template <typename DT = T>
  DT* mutable_gpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) {
    return reinterpret_cast<DT*>(mem_->mutable_gpu_data()) + offset(n, c, h, w);
  }
  template <typename DT = T>
  const DT* gpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) const {
    return reinterpret_cast<const DT*>(mem_->gpu_data()) + offset(n, c, h, w);
  }
  template <typename DT = T>
  DT* mutable_cpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) {
    return reinterpret_cast<DT*>(mem_->mutable_cpu_data()) + offset(n, c, h, w);
  }
  template <typename DT = T>
  const DT* cpu_data(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) const {
    return reinterpret_cast<const DT*>(mem_->cpu_data()) + offset(n, c, h, w);
  }
private:
  inline int offset(unsigned n = 0, unsigned c = 0, unsigned h = 0, unsigned w = 0) const {
    return w + w_ * (h + h_ * (c + c_ * n));
  }
  
  std::shared_ptr<SyncedMemory> mem_;

  unsigned n_ = 0, c_ = 0, h_ = 0, w_ = 0;
  size_t size_ = 0;
  size_t count_ = 0;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T>& t) {
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

template<typename T>
inline Tensor<T>::Tensor(unsigned n, unsigned c, unsigned h, unsigned w) :
  n_(n), c_(c), h_(h), w_(w), count_(n*c*h*w), size_(n*c*h*w * sizeof(T)) {
  mem_.reset(new SyncedMemory(size_));
}

// WIP, TODO: refer to cv::Mat::create() [core\src\matrix.cpp]
// create a new memory block for access (usually as an output arg)
template<typename T>
inline void Tensor<T>::create(unsigned n, unsigned c, unsigned h, unsigned w) {
  int count = n * c * h * w;
  int size = count * sizeof(T);
  if (size > size_)
    mem_.reset(new SyncedMemory(size));
  count_ = count;
  size_ = size;
  n_ = n; c_ = c; h_ = h; w_ = w;
}

// WIP: reshape only, without changing the underlying data
template<typename T>
inline void Tensor<T>::reshape(unsigned n, unsigned c, unsigned h, unsigned w) {
  int count = n * c * h * w;
  CHECK_EQ(count, count_) << "Total elements dismatch";
  n_ = n; c_ = c; h_ = h; w_ = w;
}


// perform shallow/deep copy
template<typename T>
inline Tensor<T>::Tensor(const Tensor & copy, bool deep_copy) : Tensor(copy.n(), copy.c(), copy.h(), copy.w()) {
  if (deep_copy) {
    if (copy.mem_.get()->size() > 0) // mem_ inited
      CHECK_CUDA(cudaMemcpy(mutable_gpu_data(), copy.gpu_data(), copy.size(), cudaMemcpyDeviceToDevice));
  }
  else {
    mem_.reset(copy.mem_.get());
  }
}

template<typename T>
inline Tensor<T>& Tensor<T>::reshape(unsigned n, unsigned c, unsigned h, unsigned w) const
{
  NOT_IMPLEMENTED; // throw
  // TODO: insert return statement here
  Tensor<T> t = *this; // create a copy of this tensor
  t.reshape(n, c, h, w); // then change its shape without affecting this tensor
  return t;
}

template<typename T>
Tensor<T> Tensor<T>::deep_copy() const {
  // Something wrong here, probably because t is deallocated on return
  //NOT_IMPLEMENTED;
  Tensor<T> t(n_, c_, h_, w_);
  CHECK_CUDA(cudaMemcpy(t.mutable_gpu_data(), this->gpu_data(), this->size_, cudaMemcpyDeviceToDevice));
  return t;
}

template<typename T>
inline std::string Tensor<T>::property_string() {
  std::stringstream s_;
  s_ << "<Tensor object of shape ("\
    << n_ << ',' << c_ << ',' << h_ << ',' << w_ << ")>";
  return s_.str();
}
