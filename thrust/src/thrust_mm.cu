#include "argparse.hpp"
#include "perfostep.hpp"
#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>
#include <sys/time.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

struct dp {
  float *A, *B;
  int m, n, r;
  dp(float *_A, float *_B, int _m, int _n, int _r)
      : A(_A), B(_B), m(_m), n(_n), r(_r){};
  __host__ __device__ float operator()(size_t idx) {
    float sum = 0.0f;
    int row = idx / r;
    int col = idx - (row * r); // cheaper modulo
    for (int i = 0; i < m; i++)
      sum += A[col + row * i] * B[col + row * i];
    return sum;
  }
};

int main(int argc, char *argv[]) {
  print_device_info();

  argparse::ArgumentParser program("test");
  program.add_argument("-m", "--m")
      .help("Number of rows in matrix A and C.")
      .scan<'u', size_t>();
  program.add_argument("-n", "--n")
      .help("Number of columns in matrix B and C.")
      .scan<'u', size_t>();
  program.add_argument("-k", "--k")
      .help("Number of columns in matrix A and rows in matrix B.")
      .scan<'u', size_t>();
  program.add_argument("-p", "--precision")
      .help("Precision of the matrix elements.")
      .default_value(std::string("fp16"));
  program.add_argument("-co", "--thrust_csv")
      .help("Path to the CSV file to store the performance results of the "
            "Thrust GEMM.")
      .default_value(std::string());

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  size_t m = program.get<size_t>("--m");
  size_t k = program.get<size_t>("--k");
  size_t n = program.get<size_t>("--n");

  const std::string csv_file = program.get<std::string>("--thrust_csv");
  const std::string precision = program.get<std::string>("--precision");

  thrust::default_random_engine rng(1337);
  thrust::uniform_int_distribution<int> dist;
  thrust::host_vector<int> a_vec(n * m);
  thrust::host_vector<int> b_vec(m * k);

  thrust::generate(a_vec.begin(), a_vec.end(), [&] { return dist(rng); });
  thrust::generate(b_vec.begin(), b_vec.end(), [&] { return dist(rng); });

  // data setup
  thrust::device_vector<float> a_dev(a_vec);
  thrust::device_vector<float> b_dev(b_vec);
  thrust::device_vector<float> c_dev(n * k);

  cudaDeviceSynchronize();
  // method 2
  // let's pretend that data is (already) transposed for efficient memory access
  // by thrust
  // therefore each dot-product is formed using a column of data and a column of
  // other
  Perfostep step;
  step.Start("Thrust GEMM");
  thrust::transform(thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(n * k), c_dev.begin(),
                    dp(thrust::raw_pointer_cast(a_dev.data()),
                       thrust::raw_pointer_cast(b_dev.data()), m, n, k));
  cudaDeviceSynchronize();
  const double latency = step.Stop();

  static const std::string BANDWIDTH = "Effective Bandwidth (GB/s)";
  static const std::string TFLOPS = "Effective TFLOPS";
  // register the performance results.
  float const thrust_effective_bandwidth{
      compute_effective_bandwidth<float>(m, n, k, latency)};
  float const thrust_effective_tflops{
      compute_effective_tflops(m, n, k, latency)};
  // add to perfostep
  const ColumnNames cuda_cols = {
      {"k", std::to_string(k)},
      {"m", std::to_string(m)},
      {"n", std::to_string(n)},
      {"Precision", precision},
      {BANDWIDTH, std::to_string(thrust_effective_bandwidth)},
      {TFLOPS, std::to_string(thrust_effective_tflops)}};

  step.UpdateCols(cuda_cols);

  // print the performance results.
  step.Report();

  step.PrintToCSV(csv_file);
}