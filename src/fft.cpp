#include "fft.h"

// HASH: gpu_memory -> corresponding handle
unordered_map<void*, cufftHandle> stored_handles_;


// legacy API
// EXP: only R2C is implemented, TODO: add other types, introduce static engine
cufftHandle getOrCreateHandle(const Tensor<float>& src, const Tensor<cuComplex>& dst) {
  void* ptr = (void*)src.gpu_data();
  if (stored_handles_.count(ptr) == 0) {
    cufftHandle plan;
    int n[] = { src.h(), src.w() };
    CHECK_CUFFT(cufftPlanMany(&plan, 2, n,
      NULL, 1, src.h() * src.w(),
      NULL, 1, dst.h() * dst.w(),
      CUFFT_R2C, src.n() * src.c()));
    stored_handles_[ptr] = plan;
  }
  return stored_handles_[ptr];
}

// cufft_handle version 1.0, in replace of getOrCreateHandle(EXP)
template <typename ST, typename DT>
cufftHandle cufft_handle(const Tensor<ST>& src, const Tensor<DT>& dst, cufftType type) {
  void* ptr = (void*)src.gpu_data();
  if (stored_handles_.count(ptr) == 0) {
    // create fft plan (lazy)
    cufftHandle plan;

    int n[2] = { 0 };
    switch (type) {
    case CUFFT_C2C:
      n[0] = src.h(); n[1] = src.w();
      CHECK_CUFFT(cufftPlanMany(&plan, 2, n,
        NULL, 1, src.h() * src.w(),
        NULL, 1, dst.h() * dst.w(),
        CUFFT_C2C, src.n() * src.c()));
      break;
    case CUFFT_R2C:
      n[0] = src.h(); n[1] = src.w();
      CHECK_CUFFT(cufftPlanMany(&plan, 2, n,
        NULL, 1, src.h() * src.w(),
        NULL, 1, dst.h() * dst.w(),
        CUFFT_R2C, src.n() * src.c()));
      break;
    case CUFFT_C2R:
      // this is a special case since we cannot determine the spatial
      // size based on `src` only, so `dst` is required as well
      n[0] = dst.h(); n[1] = dst.w();
      CHECK_CUFFT(cufftPlanMany(&plan, 2, n,
        NULL, 1, src.h() * src.w(),
        NULL, 1, dst.h() * dst.w(),
        CUFFT_C2R, src.n() * src.c()));
      break;
    default:
      break;
    }
    stored_handles_[ptr] = plan;
  }
  return stored_handles_[ptr];
}

std::ostream &operator<< (std::ostream &os, const cufftComplex &d) {
  auto c = reinterpret_cast<const std::complex<float>*>(&d);
  os << *c;
  return os;
}

void fft1d(Tensor<cufftComplex>& src, Tensor<cufftComplex>& dst) {
  cufftHandle plan;
  int n = src.w();
  CHECK_CUFFT(cufftPlan1d(&plan, n,
    CUFFT_C2C, src.n() * src.c() * src.w()));
  CHECK_CUFFT(
    cufftExecC2C(plan, src.mutable_gpu_data(), dst.mutable_gpu_data(), CUFFT_FORWARD));
  CHECK_CUFFT(cudaDeviceSynchronize());
  CHECK_CUFFT(cufftDestroy(plan));
}

void fft2d(Tensor<cufftComplex>& src, Tensor<cufftComplex>& dst) {
  cufftHandle plan;
  int n[] = { src.h(), src.w() };
  CHECK_CUFFT(cufftPlanMany(&plan, 2, n,
    NULL, 1, src.h() * src.w(),
    NULL, 1, dst.h() * dst.w(),
    CUFFT_C2C, src.n() * src.c()));
  CHECK_CUFFT(
    cufftExecC2C(plan, src.mutable_gpu_data(), dst.mutable_gpu_data(), CUFFT_FORWARD));
  CHECK_CUFFT(cudaDeviceSynchronize());
  CHECK_CUFFT(cufftDestroy(plan));
}

void ifft2d(Tensor<cufftComplex>& src, Tensor<cufftComplex>& dst, bool scale) {
  cufftHandle plan;
  int n[] = { src.h(), src.w() };
  CHECK_CUFFT(cufftPlanMany(&plan, 2, n,
    NULL, 1, src.h() * src.w(),
    NULL, 1, dst.h() * dst.w(),
    CUFFT_C2C, src.n() * src.c()));
  CHECK_CUFFT(
    cufftExecC2C(plan, src.mutable_gpu_data(), dst.mutable_gpu_data(), CUFFT_INVERSE));
  if (scale) {
    float a = 1.f / (dst.h() * dst.w());
    CHECK_CUBLAS(cublasCsscal_v2(cublas_handle(), dst.count(), &a, dst.mutable_gpu_data(), 1));
  }
  CHECK_CUFFT(cudaDeviceSynchronize());
  CHECK_CUFFT(cufftDestroy(plan));
}

void fft2d(Tensor<float>& src, Tensor<cufftComplex>& dst) {
  cufftHandle plan;
  int n[] = { src.h(), src.w() };
  CHECK_CUFFT(cufftPlanMany(&plan, 2, n,
    NULL, 1, src.h() * src.w(),
    NULL, 1, dst.h() * dst.w(),
    CUFFT_R2C, src.n() * src.c()));
  CHECK_CUFFT(
    cufftExecR2C(plan, src.mutable_gpu_data(), dst.mutable_gpu_data()));
  CHECK_CUFFT(cudaDeviceSynchronize());
  CHECK_CUFFT(cufftDestroy(plan));
}

Tensor<cufftComplex> fft2d(Tensor<float>& src) {
  Tensor<cufftComplex> dst(src.n(), src.c(), src.h(), src.w() / 2 + 1);
  fft2d(src, dst);
  return dst;
}

void ifft2d(Tensor<cufftComplex>& src, Tensor<float>& dst) {
  cufftHandle plan;
  // NOTE: the n[] should take the size of real mat (i.e. src)
  int n[] = { dst.h(), dst.w() };
  CHECK_CUFFT(cufftPlanMany(&plan, 2, n,
    NULL, 1, src.h() * src.w(),
    NULL, 1, dst.h() * dst.w(),
    CUFFT_C2R, src.n() * src.c()));
  CHECK_CUFFT(
    cufftExecC2R(plan, src.mutable_gpu_data(), dst.mutable_gpu_data()));
  float a = 1.f / (dst.h() * dst.w());
  CHECK_CUBLAS(cublasSscal_v2(cublas_handle(), dst.count(), &a, dst.mutable_gpu_data(), 1));
  CHECK_CUFFT(cudaDeviceSynchronize());
  CHECK_CUFFT(cufftDestroy(plan));
}


// TODO: add auto created `dst` tensor with `create()` method
void fft2_planed(Tensor<cufftComplex>& src, Tensor<cufftComplex>& dst) {
  cufftHandle plan = cufft_handle(src, dst, CUFFT_C2C);
  CHECK_CUFFT(
    cufftExecC2C(plan, src.mutable_gpu_data(), dst.mutable_gpu_data(), CUFFT_FORWARD));
  CHECK_CUFFT(cudaDeviceSynchronize());
}

// This is an optimized version of fft2d, which reuses previously inited
// cufftPlan object. This function is well suited in repeated execution
// on the same src Tensor
void fft2_planed(Tensor<float>& src, Tensor<cufftComplex>& dst) {
  cufftHandle plan = cufft_handle(src, dst, CUFFT_R2C);
  CHECK_CUFFT(
    cufftExecR2C(plan, src.mutable_gpu_data(), dst.mutable_gpu_data()));
  CHECK_CUFFT(cudaDeviceSynchronize());
}

void ifft2_planed(Tensor<cufftComplex>& src, Tensor<cufftComplex>& dst, bool scale) {
  // actually we can reuse the forward plan, so pass `dst` as `src`
  cufftHandle plan = cufft_handle(dst, src, CUFFT_C2C);
  CHECK_CUFFT(
    cufftExecC2C(plan, src.mutable_gpu_data(), dst.mutable_gpu_data(), CUFFT_INVERSE));
  CHECK_CUFFT(cudaDeviceSynchronize());
  if (scale) {
    float a = 1.f / (dst.h() * dst.w());
    CHECK_CUBLAS(cublasCsscal_v2(cublas_handle(), dst.count(), &a, dst.mutable_gpu_data(), 1));
  }
}
void ifft2_planed(Tensor<cufftComplex>& src, Tensor<float>& dst, bool scale) {
  cufftHandle plan = cufft_handle(src, dst, CUFFT_C2R);
  CHECK_CUFFT(
    cufftExecC2R(plan, src.mutable_gpu_data(), dst.mutable_gpu_data()));
  CHECK_CUFFT(cudaDeviceSynchronize());
  if (scale) {
    float a = 1.f / (dst.h() * dst.w());
    CHECK_CUBLAS(cublasSscal_v2(cublas_handle(), dst.count(), &a, dst.mutable_gpu_data(), 1));
  }
}
