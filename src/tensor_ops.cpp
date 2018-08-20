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

// A -> P : projection matrix
// B -> F : feature tensor
// TransA : true
// TransB : false
// M, N, K is now PC, N, C
// PF : projected feature
void gemm_feature_projection(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int PC, const int N, const int C,
  const float alpha, const float* P, const float* F, const float beta,
  float* PF) {
  // Note that cublas follows fortran order.
  CHECK_EQ(TransA, CblasTrans);
  CHECK_EQ(TransB, CblasNoTrans);
  // int lda = (TransA == CblasNoTrans) ? K : M;
  int lda = C; // NOTE: lda is still C, since we don't slice Tensor P (ref pca.h)
  // int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldb = N; // represents the spatial area of feature map
  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  // So we switch the order of A and B
  CHECK_CUBLAS(cublasSgemm(cublas_handle(), cuTransB, cuTransA,
    N, PC, C, &alpha, F, ldb, P, lda, &beta, PF, N));
}

void feature_projection(const Tensor<float>& P, const Tensor<float>& F, Tensor<float>& PF, const int compressed_dim) {
  CHECK_EQ(P.n() * P.c(), 1);
  CHECK_EQ(F.n(), 1);
  CHECK_EQ(P.h(), P.w());
  CHECK_EQ(P.h(), F.c());
  PF.create(F.n(), compressed_dim, F.h(), F.w());
  gemm_feature_projection(CblasTrans, CblasNoTrans,
    compressed_dim, F.h() * F.w(), F.c(), 1, P.gpu_data(), F.gpu_data(), 0, PF.mutable_gpu_data());
  // TODO: if F is cuComplex, just x2.
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
