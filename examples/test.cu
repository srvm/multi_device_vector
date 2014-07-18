#include <ctime>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <multi_device_vector.h>

static constexpr int N = 901;

template<typename _iterator, typename T>
__global__ void test_kernel(_iterator begin, _iterator end, T value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  *(begin + tid) = value;
}

int main() {
  using value_t = float;

  thrust::host_vector<value_t> h_vector(N, 0);

  std::srand(time(NULL));
  thrust::generate(h_vector.begin(), h_vector.end(),
      []() { return value_t(std::rand()) / value_t(RAND_MAX); });

  thrust::multi_device_vector<value_t> v1(h_vector.begin(), h_vector.end());
  thrust::multi_device_vector<value_t> v2(N, 1.0f);

  printf("h_vector[0] = %f\n", h_vector[0]);

  test_kernel<<<1, 2>>>(v1.begin(), v1.end(), 2.0f);

  thrust::host_vector<value_t> r1(v1.begin(), v1.end());
  thrust::host_vector<value_t> r2(v2.begin(), v2.end());

  printf("v1[0]: %f\n", r1[0]);
  printf("v2[0]: %f\n", r2[0]);
  
  // Copy everything in v1 to v2
  thrust::copy(v1.begin(), v1.end(), v2.begin());

  printf("After copy\n");
  thrust::host_vector<value_t> r3(v2.begin(), v2.end());

  printf("v2[0]: %f\n", r3[0]);

  return 0;
}
