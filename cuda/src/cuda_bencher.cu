#include "perfostep.hpp"
#include "checks.cuh"
#include "cuda_runtime.h"
#include "logging.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <vector>
#include "parser.hpp"

namespace cuda_vector {

__global__ void addVectorsInto(float *result, float *a, float *b, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < N; i += stride) {
    result[i] = a[i] * b[i];
  }
}

__global__ void checkElementsAre(float *target, float *a, float *b, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < N; i += stride) {
    if (a[i] * b[i] != target[i]) {
      printf("FAIL: %f * %f != %f\n", a[i], b[i], target[i]);
      assert(0);
    }
  }
}


void launch_cuda_kernel(const size_t &num_elements) {
  // CUDA Managed
  // Memory
  logging::banner("Testing Raw Cuda");

  float *a_d, *b_d, *c_d;

  Perfostep bencher;
  bencher.start();

  std::vector<float> a(num_elements);
  std::vector<float> b(num_elements);
  std::vector<float> c(num_elements);

  // size of vector a in megabytes

  std::mt19937 gen32;
  std::generate(a.begin(), a.end(), gen32);
  std::generate(b.begin(), b.end(), gen32);

  auto step1 = bencher.stop();
  bencher.report("Allocated on CPU to get copied to GPU");
  bencher.start();

  CHECK_CUDA_ERROR(cudaMallocManaged(&a_d, num_elements * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged(&b_d, num_elements * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged(&c_d, num_elements * sizeof(float)));
  // Copy data to device
  CHECK_CUDA_ERROR(cudaMemcpy(a_d, a.data(), num_elements * sizeof(float),
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(b_d, b.data(), num_elements * sizeof(float),
                              cudaMemcpyHostToDevice));

  cudaDeviceSynchronize();
  auto step2 = bencher.stop();
  bencher.report("Allocated on GPU");
  bencher.start();

  int gridSize = 256;
  int blockSize =
      1024; //  std::min(1024, (num_elements + gridSize - 1) / gridSize);

  addVectorsInto<<<gridSize, blockSize>>>(c_d, a_d, b_d, num_elements);
  CHECK_LAST_CUDA_ERROR();
  cudaDeviceSynchronize();

  auto step3 = bencher.stop();
  bencher.report("Launched Kernel");
  bencher.start();

  bencher.report((step1 + step2 + step3), "Total time for raw cuda");

  checkElementsAre<<<gridSize, blockSize>>>(c_d, a_d, b_d, num_elements);
  CHECK_LAST_CUDA_ERROR();
  cudaDeviceSynchronize();

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  cudaDeviceReset();
}

} // namespace cuda_vector



int main(int argc, char **argv) {

  CommandLineParser parser(argc, argv);

  const long size_in_mb = parser.getValueAsLong("n");
  int run_on_cpu = parser.getValueAsInt("cpu");
  int run_on_gpu = parser.getValueAsInt("gpu");
  int run_on_thrust_cpu = parser.getValueAsInt("t_cpu");
  int run_on_thrust_on_dev = parser.getValueAsInt("t_dev");
  int run_on_thrust_gpu = parser.getValueAsInt("t_gpu");
}