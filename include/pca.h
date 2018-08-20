#pragma once

#include "tensor.h"
#include "cuda_helper.h"

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
