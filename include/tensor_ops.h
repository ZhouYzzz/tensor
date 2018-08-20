#pragma once

#include "tensor.h"
#include "cuda_helper.h"

enum CBLAS_TRANSPOSE { CblasNoTrans, CblasTrans};

/* LOW LEVEL APIS ( BARE POINTER ) */
void gemm(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  const float alpha, const float* A, const float* B, const float beta,
  float* C);

/* HIGH LEVEL APIS ( TENSOR ) */
void matmul(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C,
  CBLAS_TRANSPOSE TransA = CblasNoTrans, CBLAS_TRANSPOSE TransB = CblasNoTrans);
