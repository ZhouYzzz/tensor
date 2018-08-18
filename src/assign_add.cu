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
    int j = w + h * src_ld + b * src_stride;
    dst[j] = dst[j] + src[i];
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