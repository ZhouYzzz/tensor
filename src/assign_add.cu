#include "assign_add.h"

__global__ void assign_add_2d_float_kernel(
  int n,
  const float* src, int src_ld, int src_stride,
  float* dst, int dst_ld, int dst_stride,
  int height, int width, int howmany) {
  CUDA_KERNEL_LOOP(i, n) {
    int w = i % width;
    int h = (i / width) % height;
    int b = i / (height * width);
    int si = w + h * src_ld + b * src_stride;
    int di = w + h * dst_ld + b * dst_stride;
    dst[di] = dst[di] + src[si];
  };
}

// Within ROI(height, width): dst = dst + src
void assignAdd2DImpl(
  const float* src, int src_ld, int src_stride,
  float* dst, int dst_ld, int dst_stride,
  int height, int width, int howmany) {
  
  int n = height * width * howmany;
  assign_add_2d_float_kernel<<<CUDA_NUM_BLOCKS(n), CUDA_NUM_THREADS>>>(
    n, src, src_ld, src_stride, dst, dst_ld, dst_stride, height, width, howmany
  );

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

template <>
void assignAdd2D(Tensor<float>& src, Tensor<float>& dst) {
  CHECK_LE(src.n(), dst.n());
  CHECK_LE(src.c(), dst.c());
  // TODO: broadcast support
  assignAdd2DImpl(
    src.gpu_data(),
    src.w(),
    src.w() * src.h(),
    dst.mutable_gpu_data(),
    dst.w(),
    dst.w() * dst.h(),
    src.h(),
    src.w(),
    src.n() * src.c()
  );
}

template <>
void assignAdd2D(Tensor<cuComplex>& src, Tensor<cuComplex>& dst) {
  CHECK_LE(src.n(), dst.n());
  CHECK_LE(src.c(), dst.c());
  // TODO: broadcast support
  assignAdd2DImpl(
    src.gpu_data<float>(),
    src.w() * 2,
    src.w() * src.h() * 2,
    dst.mutable_gpu_data<float>(),
    dst.w() * 2,
    dst.w() * dst.h() * 2,
    src.h(),
    src.w() * 2,
    src.n() * src.c()
  );
}
