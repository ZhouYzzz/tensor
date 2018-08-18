/* !! ABORTED !! */
/*
  This is an experimental cuda file, which implements a test version
  of memset and assign_add.

  assign_add is now availble in assign_add.h
  
*/
#include "common.h"

void f_memset(int n, float a, float *x);

void f_memcpyadd2D(float* dst, int old, int ostride, const float* src, int iw, int ih, int howmany);

/*
  f_memset(35, 2, xx.mutable_gpu_data());

  for (int i = 0; i < 2*3; i++) {
    yy.mutable_cpu_data()[i] = 1;
  }

  CHECK_CUDA( cudaMemcpy2D(
    xx.mutable_gpu_data(0, 0, 2, 2),
    7*sizeof(float),
    yy.gpu_data(),
    3*sizeof(float),
    3*sizeof(float),
    2,
    cudaMemcpyDeviceToDevice
  ));
  for (int i = 0; i < 100; i++)
    f_memcpyadd2D(
      xx.mutable_gpu_data(0, 0, 0, 0),
      7,
      7*5,
      yy.gpu_data(),
      3,
      2,
      1
    );

  // xx.cpu_data();
  cout << xx << endl;
  cout << yy << endl;
  */

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
