Multi-Device Vector
===================

Implements a vector type that spans multiple NVIDIA GPUs.

### Pre-Reqs

* CUDA Toolkit 6.5 or above

### Installation

This library is header-only and does not require installation.

### Sample Usage

```c++
...

// Declare vector on host
thrust::host_vector<value_t> h_vector(1024, 0);

// Fill with random values
std::srand(time(NULL));
thrust::generate(h_vector.begin(), h_vector.end(),
    []() { return value_t(std::rand()) / value_t(RAND_MAX); });

// Declare vector that spans all available GPUs
thrust::multi_device_vector<value_t> v1(h_vector.begin(), h_vector.end());

...

```
