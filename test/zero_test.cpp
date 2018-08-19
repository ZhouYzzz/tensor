#include <gtest/gtest.h>

#include "cuda_helper.h"

TEST(Zero, ZeroTest) {
  ASSERT_EQ(0, 0);
}

TEST(Zero, CudaEnvironment) {
  auto cublas_h = cublas_handle();
  auto cudnn_h = cudnn_handle();
  ASSERT_EQ(cublas_h, cublas_handle());
  ASSERT_EQ(cudnn_h, cudnn_handle());
}
