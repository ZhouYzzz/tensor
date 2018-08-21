#include "pca.h"

// WIP: a low level routine for svd
// params:
// A:(m, n), S:(1, min(m, n)), U:(m, min(m, n)), V:(n, min(m, n))
// NOTE: On exit, the contents of A are destroyed.
void gesvdj(int m, int n, float * A, float * S, float * U, float * V) {
  cusolverDnSgesvdj;
  cusolverDnHandle_t handle = cusolverDn_handle();
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  int econ = 1;
  int lda = m;
  int ldu = m;
  int ldv = n;
  float* d_work = NULL;
  int lwork = 0;
  int* d_info = NULL;
  int info = 0;
  gesvdjInfo_t params;
  const double tol = 1.e-7;
  CHECK_CUSOLVER(cusolverDnCreateGesvdjInfo(&params));
  CHECK_CUSOLVER(cusolverDnXgesvdjSetTolerance(params, tol));
  CHECK_CUSOLVER(cusolverDnSgesvdj_bufferSize(handle, jobz, econ,
    m, n, A, lda, S, U, ldu, V, ldv, &lwork, params));
  CHECK_CUDA(cudaMalloc(&d_work, sizeof(float)*lwork));
  CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
  CHECK_CUSOLVER(cusolverDnSgesvdj(handle, jobz, econ,
    m, n, A, lda, S, U, ldu, V, ldv, d_work, lwork, d_info, params));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_EQ(info, 0) << "gesvdj fail";
  CHECK_CUDA(cudaFree(d_work));
  CHECK_CUDA(cudaFree(d_info));
  CHECK_CUSOLVER(cusolverDnDestroyGesvdjInfo(params));
}

// Econ version of SVD
// A is represented in fortran format, symmetric matrix doesn't matter, but otherwise not tested
// TODO: test general case
void SVD_econ(Tensor<float>& A, Tensor<float>& S, Tensor<float>& U, Tensor<float>& V) {
  CHECK_EQ(A.n() * A.c(), 1) << "Only 2D Matrix is supported";
  int m = A.w(); // m is leading dim, since svd takes fortran format, m == A.w() in C++
  int n = A.h();
  int d = fmin(m, n); // size of second dim
  S.create(1, 1, 1, d);
  U.create(1, 1, d, m);
  V.create(1, 1, d, n);
  gesvdj(m, n,
    A.mutable_gpu_data(),
    S.mutable_gpu_data(),
    U.mutable_gpu_data(),
    V.mutable_gpu_data());
}

// The SVD used in ECO applies on symmetric matrix (X' * X)
// This makes things much more easier than general case.
void SVD(Tensor<float>& A, Tensor<float>& S, Tensor<float>& U, Tensor<float>& V) {
  CHECK_EQ(A.n() * A.c(), 1); // TODO: multidim support?
  CHECK_EQ(A.h(), A.w()) << "Only symmetric mat is supported now"; // TODO: general case
                                                                   // Note that cuSolver also works in column major order

  int M = A.h();
  int N = A.w();

  S.create(1, 1, 1, N);
  U.create(1, 1, N, N);
  V.create(1, 1, N, N);

  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  gesvdjInfo_t gesvdj_params = NULL;
  const double tol = 1.e-7;
  int lwork = 0;       /* size of workspace */

                       /* step 2: configuration of gesvdj */
  status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  /* default value of tolerance is machine zero */
  status = cusolverDnXgesvdjSetTolerance(
    gesvdj_params,
    tol);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  int *info = NULL;

  CHECK_CUSOLVER(
    cusolverDnSgesvdj_bufferSize(
      cusolverDn_handle(),
      CUSOLVER_EIG_MODE_VECTOR,
      1,
      N,
      N,
      A.gpu_data(),
      N,
      S.gpu_data(),
      U.gpu_data(),
      N,
      V.gpu_data(),
      N,
      &lwork,
      gesvdj_params
    )
  );
  LOG(INFO) << lwork;

  float* d_work;
  CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(double)*lwork));
  CUSOLVER_STATUS_INVALID_VALUE;

  CHECK_CUSOLVER(
    cusolverDnSgesvdj(
      cusolverDn_handle(),
      CUSOLVER_EIG_MODE_VECTOR,
      1, // econ = ON
      N,
      N,
      A.mutable_gpu_data(),
      N,
      S.mutable_gpu_data(),
      U.mutable_gpu_data(),
      N,
      V.mutable_gpu_data(),
      N,
      d_work,
      lwork,
      info,
      gesvdj_params
    )
  );
  CHECK_CUDA(cudaFree(d_work));
  //LOG(INFO) << *info;
}
