#pragma once

#include "common.h"
#include "tensor.h"

void assignAdd2DImpl(
  const float* src, int src_ld, int src_stride,
  float* dst, int dst_ld, int dst_stride,
  int height, int width, int howmany);

template <typename T>
void assignAdd2D(const Tensor<T>& src, Tensor<T>& dst);
