#pragma once

#include <iostream>
#include <complex>
#include <cufft.h>
#include <cublas.h>

#include <unordered_map>

#include "common.h"
#include "tensor.h"
#include "cuda_helper.h"

using std::unordered_map;

std::ostream &operator<< (std::ostream &os, const cufftComplex &d);

// Old APIs
void fft1d(Tensor<cufftComplex>& src, Tensor<cufftComplex>& dst);
void fft2d(Tensor<cufftComplex>& src, Tensor<cufftComplex>& dst);
void ifft2d(Tensor<cufftComplex>& src, Tensor<cufftComplex>& dst, bool scale = true);
void fft2d(Tensor<float>& src, Tensor<cufftComplex>& dst);
Tensor<cufftComplex> fft2d(Tensor<float>& src);
void ifft2d(Tensor<cufftComplex>& src, Tensor<float>& dst);

// planed APIs
void fft2_planed(Tensor<cufftComplex>& src, Tensor<cufftComplex>& dst);
void fft2_planed(Tensor<float>& src, Tensor<cufftComplex>& dst);
void ifft2_planed(Tensor<cufftComplex>& src, Tensor<cufftComplex>& dst, bool scale = true);
void ifft2_planed(Tensor<cufftComplex>& src, Tensor<float>& dst, bool scale = true);
