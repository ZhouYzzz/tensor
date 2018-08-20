#pragma once

#include "tensor.h"
#include "cuda_helper.h"

enum CBLAS_TRANSPOSE { CblasNoTrans, CblasTrans};

/* LOW LEVEL APIS ( BARE POINTER ) */

// matrix multiplication with C-order data arrangement
void gemm(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  const float alpha, const float* A, const float* B, const float beta,
  float* C);

/* HIGH LEVEL APIS ( TENSOR ) */
void matmul(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C,
  CBLAS_TRANSPOSE TransA = CblasNoTrans, CBLAS_TRANSPOSE TransB = CblasNoTrans);

// optimized code for ECO tracker
namespace ECO {
  // A -> P : projection matrix
  // B -> F : feature tensor
  // TransA : true
  // TransB : false
  // M, N, K is now PC, N, C
  // PF : projected feature
  void gemm_feature_projection(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int PC, const int N, const int C,
    const float alpha, const float* P, const float* F, const float beta,
    float* PF);

  void feature_projection(const Tensor<float>& P, const Tensor<float>& F, Tensor<float>& PF, const int compressed_dim);
}
