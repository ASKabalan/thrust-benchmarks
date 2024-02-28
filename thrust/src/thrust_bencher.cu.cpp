#include "logging.h"
#include "perfostep.hpp"
#include <algorithm>
#include <cstdio>
#include <thrust/copy.h>
#include <thrust/count.h>
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
#include <tuple>
#include "checks.cuh"
#include <algorithm>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include "parser.hpp"

typedef float numeric;

namespace thrust_gpu {

void run_thrust_gpu(const size_t &num_elements) {

  logging::banner("Thrust Device From thrust::host_vector");
  thrust::default_random_engine rng;
  Perfostep bencher;
  bencher.start();

  thrust::host_vector<numeric> a_h(num_elements);
  thrust::host_vector<numeric> b_h(num_elements);
  thrust::host_vector<numeric> c_h(num_elements);

  thrust::generate(a_h.begin(), a_h.end(), rng);
  thrust::generate(b_h.begin(), b_h.end(), rng);

  auto step4 = bencher.stop();
  bencher.report("Allocating on thrust::host_vector");
  bencher.start();

  thrust::device_vector<numeric> thrustDeviceA = a_h;
  thrust::device_vector<numeric> thrustDeviceB = b_h;
  thrust::device_vector<numeric> thrustDeviceC(num_elements, 0);

  auto step5 = bencher.stop();
  bencher.report(
      "Allocating on thrust::device_vector from thrust::host_vector");
  bencher.start();

  //thrust::transform(thrustDeviceA.begin(), thrustDeviceA.end(),
  //                  thrustDeviceB.begin(), thrustDeviceC.begin(),
  //                  thrust::multiplies<numeric>());

  auto step6 = bencher.stop();
  bencher.report("Thrust::transform on GPU");

  bencher.report((step4 + step5 + step6),
                 "Total time for thrust::host_vector to device to transform");
}

void run_thrust_on_dev(const size_t &num_elements) {

  thrust::default_random_engine rng;
  logging::banner("Thrust Allocate directly on device");
  Perfostep bencher;
  bencher.start();

  thrust::device_vector<numeric> stdthrustDeviceA(num_elements);
  thrust::device_vector<numeric> stdthrustDeviceB(num_elements);
  thrust::device_vector<numeric> stdthrustDeviceC(num_elements);

  thrust::generate(stdthrustDeviceA.begin(), stdthrustDeviceA.end(), rng);
  thrust::generate(stdthrustDeviceB.begin(), stdthrustDeviceB.end(), rng);

  auto step1 = bencher.stop();
  bencher.report("Allocating on thrust::device_vector");
  bencher.start();

  //thrust::transform(stdthrustDeviceA.begin(),
  //                  stdthrustDeviceA.end(), stdthrustDeviceB.begin(),
  //                  stdthrustDeviceC.begin(), thrust::multiplies<numeric>());

  SYNCHRONIZE_AND_CHECK();

  auto step2 = bencher.stop();
  bencher.report("Thrust::transform on Device");

  bencher.report((step1 + step2),
                 "Total time for std::vector to device to transform");
}

} // namespace thrust_gpu



int main(int argc, char **argv) {

  CommandLineParser parser(argc, argv);

  const long size_in_mb = parser.getValueAsLong("n");
  int run_on_cpu = parser.getValueAsInt("cpu");
  int run_on_gpu = parser.getValueAsInt("gpu");
  int run_on_thrust_cpu = parser.getValueAsInt("t_cpu");
  int run_on_thrust_on_dev = parser.getValueAsInt("t_dev");
  int run_on_thrust_gpu = parser.getValueAsInt("t_gpu");
}