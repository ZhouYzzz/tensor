#include <gtest/gtest.h>

#include "fft.h"
#include "assign_add.h"

#define CHECK_TENSOR_NEAR(T1, T2, m)\
  do {\
    CHECK_EQ(T1.count(), T2.count());\
    for (int i = 0; i < T1.count(); i++) {\
      CHECK_NEAR(T1.cpu_data()[i], T2.cpu_data()[i], m);\
    }\
  } while (0);

TEST(FFT, FFT2D) {
  Tensor<float> x(2, 1, 3, 5);
  Tensor<float> x2(2, 1, 3, 5);
  x.mutable_cpu_data()[0] = 1;
  x.mutable_cpu_data()[1] = 1;
  x.mutable_cpu_data(0, 0, 1, 0)[0] = 1;
  x.mutable_cpu_data(1, 0, 0, 0)[0] = 1;
  //cout << x << endl;
  Tensor<cuComplex> xf = fft2d(x);
  //cout << xf << endl;
  ifft2d(xf, x2);
  //cout << x2 << endl;

  CHECK_TENSOR_NEAR(x2, x, 1e-6);

  Tensor<float> y(2, 1, 3, 5);
  Tensor<float> y2(2, 1, 3, 5);
  y.mutable_cpu_data(0, 0, 1, 0)[1] = 1;
  y.mutable_cpu_data(1, 0, 1, 0)[1] = 1;
  //cout << y << endl;
  Tensor<cuComplex> yf = fft2d(y);
  //cout << yf << endl;

  assignAdd2D(xf, yf);
  assignAdd2D(x, y);
  ifft2d(yf, y2);
  //cout << y << endl;

  CHECK_TENSOR_NEAR(y, y2, 1e-6);
}

TEST(FFT, Speed) {
  Tensor<cuComplex> x(60, 64, 31, 31);
  Tensor<cuComplex> xf(60, 64, 31, 31);
  //Tensor<float> x2(60, 64, 31, 60);
  //Tensor<cuComplex> xf;
  for (int i = 0; i < 1; i++)
    fft2_planed(x, xf);
  cout << xf.property_string() << endl;
}
