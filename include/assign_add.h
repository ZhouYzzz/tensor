#pragma once

#include "common.h"
#include "tensor.h"

void assignAdd2DImpl(
  const float* src, int src_ld, int src_stride,
  float* dst, int dst_ld, int dst_stride,
  int height, int width, int howmany);

void assignAdd2D(Tensor<float>& src, Tensor<float>& dst) {
  CHECK_EQ(src.n(), dst.n()) << "N dismatch";
  CHECK_EQ(src.c(), dst.c()) << "C dismatch";
  // TODO: broadcast support
  assignAdd2DImpl(
    src.gpu_data(),
    src.w(),
    src.w() * src.h(),
    dst.mutable_gpu_data(),
    dst.w(),
    dst.w() * dst.h(),
    dst.h(),
    dst.w(),
    src.n() * src.c()
  );
}