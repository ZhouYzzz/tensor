#pragma once

#include <assert.h>
#include "tensor.h"
#include "cuda_helper.h"

namespace ECO {
  // WIP: High level API designed for ECO (feature compression)
  /*
    input:
      Tensor<float>& feature: (1, C, H, W) in C-order, reshape as a mat (C, N)

    output:
      Tensor<float>& P is the `U` mat calculated with SVD(feature*feature'), which should be (C, C)
      `U` is supposed to be symmetric
  */
  void calculate_projection_matrix(const Tensor<float>& feature, Tensor<float>& P);

  /*
    input:
      Tensor<float>& feature: (1, C, H, W) in C-order, reshape as a mat (C, N)

    Here we call matrix multiplication to get a (dim, N) output.

      projected (dim, N) = P (C, dim out of C) * feature (C, N)    .......  C-order

    That is, gemm(P, feature, projected, NT, NT) if P is (C, dim)

    However, P is now (C, C), we can take advantage of `lda' parameter, thus

            gemm(P, feature, projected, T, NT) with `lda = C; M = dim;`
  */
  void project_sample(const Tensor<float>& feature, const Tensor<float>& P, const int dim, Tensor<float>& projected);
}

// WIP: a low level routine for svd
// params:
// A:(m, n), S:(1, min(m, n)), U:(m, min(m, n)), V:(n, min(m, n))
// NOTE: On exit, the contents of A are destroyed.
void gesvdj(int m, int n, float* A, float* S, float* U, float* V);

// Econ version of SVD
// A is represented in fortran format, symmetric matrix doesn't matter, but otherwise not tested
// TODO: test general case
void SVD_econ(Tensor<float>& A, Tensor<float>& S, Tensor<float>& U, Tensor<float>& V);


// The SVD used in ECO applies on symmetric matrix (X' * X)
// This makes things much more easier than general case.
void SVD(Tensor<float>& A, Tensor<float>& S, Tensor<float>& U, Tensor<float>& V);
