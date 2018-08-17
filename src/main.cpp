#include "tensor.h"
#include "fft.h"

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

  auto xptr = reinterpret_cast<complex<float>*>(x.mutable_cpu_data());

  xptr[0] = {1, 0};
  xptr[1] = {1, 0};
  for (int i = 0; i < 9; i++) {
    cout << xptr[i] << ',';
  } cout << endl;
  
  fft2d(x, y);
  
  auto yptr = reinterpret_cast<complex<float>*>(y.mutable_cpu_data());

  for (int i = 0; i < 9; i++) {
    cout << yptr[i] << ',';
  } cout << endl;
  cout << "Hello CMake." << endl;
  return 0;
}
