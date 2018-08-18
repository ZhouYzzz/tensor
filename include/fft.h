#pragma once

#include <iostream>
#include <complex>
#include <cufft.h>

#include "tensor.h"

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
  int n[] = {src.h(), src.w()};
  CHECK_CUFFT(cufftPlanMany(&plan, 2, n,
    NULL, 1, src.h() * src.w(),
    NULL, 1, src.h() * src.w(),
    CUFFT_C2C, src.n() * src.c()));
  CHECK_CUFFT(
    cufftExecC2C(plan, src.mutable_gpu_data(), dst.mutable_gpu_data(), CUFFT_FORWARD));
  CHECK_CUFFT(cudaDeviceSynchronize());
  CHECK_CUFFT(cufftDestroy(plan));
}

void ifft2d(Tensor<cufftComplex>& src, Tensor<cufftComplex>& dst) {
  cufftHandle plan;
  int n[] = {src.h(), src.w()};
  CHECK_CUFFT(cufftPlanMany(&plan, 2, n,
    NULL, 1, src.h() * src.w(),
    NULL, 1, src.h() * src.w(),
    CUFFT_C2C, src.n() * src.c()));
  CHECK_CUFFT(
    cufftExecC2C(plan, src.mutable_gpu_data(), dst.mutable_gpu_data(), CUFFT_INVERSE));
  CHECK_CUFFT(cudaDeviceSynchronize());
  CHECK_CUFFT(cufftDestroy(plan));
}
