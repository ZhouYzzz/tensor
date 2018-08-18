#include "tensor.h"
#include "fft.h"
#include "assign_add.h"

#include <complex>

using namespace std;

int main()
{
  Tensor<float> t(1, 1, 20, 20);
  t.mutable_cpu_data()[0] = 1;
  t.gpu_data();
  cout << t.mutable_cpu_data()[0] << endl;

  cout << sizeof(cufftComplex) << ',' << sizeof(complex<float>) << endl;
  Tensor<cufftComplex> x(1, 1, 3, 3);
  Tensor<cufftComplex> y(1, 1, 3, 3);

  x.mutable_cpu_data<complex<float>>()[0] = {1, 0};
  for (int i = 0; i < 9; i++) {
    cout << x.cpu_data()[i] << ',';
  } cout << endl;
  
  fft2d(x, y);
  
  auto yptr = reinterpret_cast<complex<float>*>(y.mutable_cpu_data());

  for (int i = 0; i < 9; i++) {
    cout << y.cpu_data()[i] << ',';
  } cout << endl;
  cout << "Hello CMake." << endl;

  Tensor<float> xx(3, 1, 5, 7);
  Tensor<float> yy(2, 1, 2, 3);


  for (int i = 0; i < yy.count(); i++) {
    yy.mutable_cpu_data()[i] = i;
  }
  for (int i = 0; i < xx.count(); i++) {
    xx.mutable_cpu_data()[i] = i;
  }

  //assignAdd2DImpl(
  //  yy.gpu_data(),
  //  yy.w(),
  //  yy.w() * yy.h(),
  //  xx.mutable_gpu_data(),
  //  xx.w(),
  //  xx.w() * xx.h(),
  //  yy.h(),
  //  yy.w(),
  //  yy.n() * yy.c()
  //);

  assignAdd2D(yy, xx);

  cout << xx << endl;
  cout << yy << endl;

  getchar();

  return 0;
}
