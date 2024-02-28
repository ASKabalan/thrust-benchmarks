#include "kernels.h"
#include "kernels_gpu.h"
#include <iostream>

#include "parser.hpp"
#include <string_view>
#include <vector>

int main(int argc, char **argv) {

  CommandLineParser parser(argc, argv);

  const long size_in_mb = parser.getValueAsLong("n");
  int run_on_cpu = parser.getValueAsInt("cpu");
  int run_on_gpu = parser.getValueAsInt("gpu");
  int run_on_thrust_cpu = parser.getValueAsInt("t_cpu");
  int run_on_thrust_on_dev = parser.getValueAsInt("t_dev");
  int run_on_thrust_gpu = parser.getValueAsInt("t_gpu");

  // Create N as number of elements of floats from size_in_m
  const long N = (size_in_mb * 1024 * 1024) / sizeof(float);

  // Compute vector size in gb
  std::cout << "Vector size: "
            << static_cast<double>(N * sizeof(float)) / (1024 * 1024) << " MB"
            << std::endl;

  std::cout << "Run on CPU: " << run_on_cpu << std::endl;

  /**********************/
  // ** Host Functions **
  /**********************/
  if (run_on_cpu)
    cpu_vector::run_on_cpu(N);
  // ***************
  // ** Pure CUDA **
  // ***************
  if (run_on_gpu)
    cuda_vector::launch_cuda_kernel(N);
  // ***************
  // ** Thrust for CPU **
  // ***************
  if (run_on_thrust_cpu)
    thrust_cpu::run_thrust_cpu(N);
  // ***************
  // ** Thrust for GPU **
  // ***************
  if (run_on_thrust_gpu)
    thrust_gpu::run_thrust_gpu(N);
  // ***************
  // ** Thrust for GPU on dev directly **
  // ***************
  if (run_on_thrust_on_dev)
    thrust_gpu::run_thrust_on_dev(N);

  return 0;
}
