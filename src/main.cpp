#include "tensor.h"

using namespace std;

int main()
{
  Tensor<float> t(1, 1, 20, 20);
  t.gpu_data();
  t.mutable_gpu_data();
  t.cpu_data();
  t.mutable_cpu_data();
  cout << "Hello CMake." << endl;
  getchar();
  return 0;
}
