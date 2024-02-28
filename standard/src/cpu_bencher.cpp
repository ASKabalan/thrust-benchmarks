#include "logging.h"
#include "perfostep.hpp"
#include "parser.hpp"
#include <algorithm>
#include <cstdio>
#include <execution>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <vector>

namespace cpu_vector {

void run_on_cpu(const size_t &num_elements) {

  logging::banner("Testing on CPU using std library");

  Perfostep step;
  step.start();

  std::vector<float> a(num_elements);
  std::vector<float> b(num_elements);
  std::vector<float> c(num_elements);

  std::mt19937 gen32;
  std::generate(a.begin(), a.end(), gen32);
  std::generate(b.begin(), b.end(), gen32);

  auto step1 = step.stop();
  step.report("Allocating using std::generate on CPU");
  step.start();

  std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(),
                 c.begin(), std::multiplies<float>());

  auto step2 = step.stop();
  step.report("Adding using std::transform on CPU");
  step.start();

  step.report((step2 + step1), "Total time for CPU");
}

} // namespace cpu_vector

int main(int argc, char **argv) {

  CommandLineParser parser(argc, argv);

  const long size_in_mb = parser.getValueAsLong("n");
  int run_on_cpu = parser.getValueAsInt("cpu");
  int run_on_gpu = parser.getValueAsInt("gpu");
  int run_on_thrust_cpu = parser.getValueAsInt("t_cpu");
  int run_on_thrust_on_dev = parser.getValueAsInt("t_dev");
  int run_on_thrust_gpu = parser.getValueAsInt("t_gpu");
}