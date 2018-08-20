#include <gtest/gtest.h>

#include <memory>
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


TEST(Zero, SyncedMem) {
  //using std::shared_ptr;
  //using std::cout;
  using namespace std;
  class TestObj {
  public:
    TestObj() {
      LOG(INFO) << this << ":allocated.";
    }
    ~TestObj() {
      LOG(INFO) << this << ":destroied.";
    }
  };

  shared_ptr<TestObj> a;
  a = make_shared<TestObj>();

  // make a copy of a
  shared_ptr<TestObj> b = a;

  a.reset(new TestObj());

  LOG(INFO) << "b still holds 1st instance";

  b.reset();

  a.reset();
}