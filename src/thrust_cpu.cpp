#include "perfostep.h"
#include "parser.h"
#include "logging.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/random.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <vector>
namespace thrust_cpu {

void run_thrust_cpu(const size_t &num_elements) {

  logging::banner("Test Thrust on CPU");

  Perfostep perfostep;
  perfostep.start();
  std::mt19937 gen32;
  thrust::host_vector<float> a_h(num_elements);
  thrust::host_vector<float> b_h(num_elements);
  thrust::host_vector<float> c_h(num_elements);

  thrust::generate(a_h.begin(), a_h.end(), gen32);
  thrust::generate(b_h.begin(), b_h.end(), gen32);

  auto step1 = perfostep.stop();
  perfostep.report("Allocating on thrust::host_vector");
  perfostep.start();

    thrust::transform(a_h.begin(), a_h.end(), b_h.begin(),
                    c_h.begin(), thrust::multiplies<float>());

  auto step2 = perfostep.stop();
  perfostep.report("Thrust::transform on CPU");

  perfostep.report((step1 + step2), "Total time for thrust CPU");

  perfostep.start();
}

} // namespace thrust_cpu



int main(int argc, char **argv) {

  CommandLineParser parser(argc, argv);

  const long size_in_mb = parser.getValueAsLong("n");
  int run_on_cpu = parser.getValueAsInt("cpu");
  int run_on_gpu = parser.getValueAsInt("gpu");
  int run_on_thrust_cpu = parser.getValueAsInt("t_cpu");
  int run_on_thrust_on_dev = parser.getValueAsInt("t_dev");
  int run_on_thrust_gpu = parser.getValueAsInt("t_gpu");
}