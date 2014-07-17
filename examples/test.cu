#include <ctime>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <multi_device_vector.h>

static constexpr int N = 901;

template<typename S>
__global__ void test_kernel(S s) {
  s[0] = 3;
  printf("%f\n", s[0]);
}

int main() {
  using value_t = float;

  thrust::host_vector<value_t> h_vector(N, 0);

  std::srand(time(NULL));
  thrust::generate(h_vector.begin(), h_vector.end(),
      []() { return value_t(std::rand()) / value_t(RAND_MAX); });

  thrust::multi_device_vector<value_t> v1(h_vector);
  thrust::multi_device_vector<value_t> v2(N, 0);

  test_kernel<<<1, 1>>>(v1);
  
  // Copy everything in v1 to v2
  //thrust::copy(v1.begin(), v1.end(), v2.begin());

  //thrust::host_vector<value_t> result(v2.begin(), v2.end());

  return 0;
}
