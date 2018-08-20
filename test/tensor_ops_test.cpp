#include <gtest/gtest.h>

#include "tensor_ops.h"
#include "tensor.h"

TEST(TensorOps, matmul) {
  /*
    0, 1     0, 1, 2    3, 4, 5
    2, 3  *  3, 4, 5 =  9, 14,19
    4, 5                15,24,33
    6, 7                21,34,47
  */
  Tensor<float> A(1, 1, 4, 2);
  Tensor<float> B(1, 1, 2, 3);
  for (int i = 0; i < A.count(); i++)
    A.mutable_cpu_data()[i] = i;
  for (int i = 0; i < B.count(); i++)
    B.mutable_cpu_data()[i] = i;
  cout << A << endl;
  cout << B << endl;
  Tensor<float> C;
  //C.create(1, 1, 3, 3);
  matmul(A, B, C);
  cout << C << endl;

  Tensor<float> D;
  matmul(A, A, D, CblasTrans, CblasNoTrans);
  cout << D << endl;
  matmul(A, A, D, CblasNoTrans, CblasTrans);
  cout << D << endl;
}
