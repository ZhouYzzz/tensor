#include "tensor_ops.h"

void gemm(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  const float alpha, const float* A, const float* B, const float beta,
  float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  // So we switch the order of A and B
  CHECK_CUBLAS(cublasSgemm(cublas_handle(), cuTransB, cuTransA,
    N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

void matmul(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB) {
  // TODO: add multidim support
  CHECK_EQ(A.n() * A.c(), 1);
  CHECK_EQ(B.n() * B.c(), 1);

  //int M = A.h();
  //int N = B.w();
  //int K = A.w();

  int M = (TransA == CblasNoTrans) ? A.h() : A.w();
  int N = (TransB == CblasNoTrans) ? B.w() : B.h();
  int KA = (TransA == CblasNoTrans) ? A.w() : A.h();
  int KB = (TransB == CblasNoTrans) ? B.h() : B.w();
  CHECK_EQ(KA, KB) << "K dim dismatch";

  int K = KA;

  //CHECK_EQ(B.h(), K) << "Dim dismatch";
  C.create(1, 1, M, N);
  gemm(TransA, TransB, M, N, K,
    1, A.gpu_data(), B.gpu_data(), 0, C.mutable_gpu_data());
}
