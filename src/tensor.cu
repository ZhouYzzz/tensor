#include "common.h"

__global__ void f_memset_kernel(int n, float a, float *x) {
  CUDA_KERNEL_LOOP(i, n) {
    x[i] = a;
  };
}

void f_memset(int n, float a, float *x) {
  f_memset_kernel<<<CUDA_NUM_BLOCKS(n), CUDA_NUM_THREADS>>>(n, a, x);
}


__global__ void f_memcpyadd2D_kernel(int n, float* dst, int old, int ostride, const float* src, int iw, int ih, int howmany) {
  CUDA_KERNEL_LOOP(i, n) {
    int w = i % iw;
    int h = (i / iw) % ih;
    int c = i / (iw * ih);
    int j = c * ostride + h * old + w;
    dst[j] = dst[j] + src[i];
  };
}


void f_memcpyadd2D(float* dst, int old, int ostride, const float* src, int iw, int ih, int howmany) {
  int n = iw * ih * howmany;
  f_memcpyadd2D_kernel<<<CUDA_NUM_BLOCKS(n), CUDA_NUM_THREADS>>>(
    n, dst, old, ostride, src, iw, ih, howmany
  );
}
