#include "common.h"
#include <iostream>

__global__ void assign_add_2d_float_kernel(
  int n,
  const float* src, int src_ld, int src_stride,
  float* dst, int dst_ld, int dst_stride,
  int height, int width, int howmany) {
  // CUDA_KERNEL_LOOP(i, n) {
  //   dst[i] = 0;
  // }
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
  std::cout << "=" << n << std::endl;
  assign_add_2d_float_kernel<<<CUDA_NUM_BLOCKS(n), CUDA_NUM_THREADS>>>(
    n, src, src_ld, src_stride, dst, dst_ld, dst_stride, height, width, howmany
  );

  //CHECK_CUDA(cudaGetLastError());
  //CHECK_CUDA(cudaDeviceSynchronize());
}

void assignAdd2D(Tensor<float>& src, Tensor<float>& dst) {
  CHECK_EQ(src.n(), dst.n()) << "N dismatch";
  CHECK_EQ(src.c(), dst.c()) << "C dismatch";
  // TODO: broadcast support
  assignAdd2DImpl(
    src.gpu_data(),
    src.w(),
    src.w() * src.h(),
    dst.mutable_gpu_data(),
    dst.w(),
    dst.w() * dst.h(),
    dst.h(),
    dst.w(),
    src.n() * src.c()
  );
}
