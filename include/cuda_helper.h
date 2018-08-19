#pragma once

#include "common.h"

cudnnHandle_t cudnn_handle();
cublasHandle_t cublas_handle();
float* cudnn_get_workspace();
void cudnn_set_workspace(size_t size);
