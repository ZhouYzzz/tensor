#include "tensor.h"
#include "fft.h"
// #include "assign_add.h"

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

  Tensor<float> xx(2, 1, 5, 7);
  Tensor<float> yy(2, 1, 2, 3);
  f_memset(35, 2, xx.mutable_gpu_data());

  for (int i = 0; i < 2*3; i++) {
    yy.mutable_cpu_data()[i] = 1;
  }

  CHECK_CUDA( cudaMemcpy2D(
    xx.mutable_gpu_data(0, 0, 2, 2),
    7*sizeof(float),
    yy.gpu_data(),
    3*sizeof(float),
    3*sizeof(float),
    2,
    cudaMemcpyDeviceToDevice
  ));
  for (int i = 0; i < 100; i++)
    f_memcpyadd2D(
      xx.mutable_gpu_data(0, 0, 0, 0),
      7,
      7*5,
      yy.gpu_data(),
      3,
      2,
      1
    );
  // for (int i = 0; i < yy.count(); i++) {
  //   yy.mutable_cpu_data()[i] = 1;
  // }
  // for (int i = 0; i < xx.count(); i++) {
  //   xx.mutable_cpu_data()[i] = 2;
  // }

  // assignAdd2D(yy, xx);

  cout << xx << endl;


  return 0;
}
