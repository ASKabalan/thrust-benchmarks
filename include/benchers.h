#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/transform.h>


namespace thrust_bencher {

void run_thrust_gpu(const size_t &n);
void run_thrust_on_dev(const size_t &n);
void run_thrust_cpu(const size_t &n);

} // namespace thrust_gpu


namespace cpu_bencher {

bool run_on_cpu(const size_t &n);

} // namespace cpu_vector

namespace cuda_bencher {

void launch_cuda_kernel(const size_t &n);
}
