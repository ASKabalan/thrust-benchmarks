////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////
#include "argparse.hpp"
#include "checks.cuh"
#include "matx.h"
#include "matx/generators/random.h"
#include "perfostep.hpp"
#include <cassert>
#include <cstdio>
#include <cuda/std/ccomplex>
#include <iostream>
#include <matx.h>
#include <stdlib.h>

template <typename T>
void run_benchmark(int m, int n, int k, const std::string &precision,
                   const std::string &cuda_csv, const std::string &cublas_csv) {
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  auto a = matx::make_tensor<T>({m, k});
  auto b = matx::make_tensor<T>({k, n});
  auto c = matx::make_tensor<T>({m, n});

  (a = matx::random<T>({m, k}, matx::UNIFORM)).run(stream);
  (b = matx::random<T>({k, n}, matx::UNIFORM)).run(stream);
  (c = matx::random<T>({m, n}, matx::UNIFORM)).run(stream);

  // Create a performance step

  Perfostep perf;
  perf.Start("MatX GEMM");
  (c = (a * b) + c).run(stream);
  cudaStreamSynchronize(stream);
  //cudaDeviceSynchronize();
  const double latency_cuda_gemm = perf.Stop();

  float const bandwith{
      compute_effective_bandwidth<T>(m, n, k, latency_cuda_gemm)};
  float const tflops{compute_effective_tflops(m, n, k, latency_cuda_gemm)};

  static const std::string BANDWIDTH = "Effective Bandwidth (GB/s)";
  static const std::string TFLOPS = "Effective TFLOPS";
  // register the performance results.
  // add to perfostep
  const ColumnNames cuda_cols = {{"k", std::to_string(k)},
                                 {"m", std::to_string(m)},
                                 {"n", std::to_string(n)},
                                 {"Precision", precision},
                                 {BANDWIDTH, std::to_string(bandwith)},
                                 {TFLOPS, std::to_string(tflops)}};
  perf.UpdateCols(cuda_cols);

  perf.Report();
}

int main(int argc, char *argv[]) {

  print_device_info();

  argparse::ArgumentParser program("test");
  program.add_argument("-m", "--m")
      .help("Number of rows in matrix A and C.")
      .scan<'i', int>();
  program.add_argument("-n", "--n")
      .help("Number of columns in matrix B and C.")
      .scan<'i', int>();
  program.add_argument("-k", "--k")
      .help("Number of columns in matrix A and rows in matrix B.")
      .scan<'i', int>();
  program.add_argument("-p", "--precision")
      .help("Precision of the matrix elements.")
      .default_value(std::string("fp16"));
  program.add_argument("-co", "--cuda_csv")
      .help("Path to the CSV file to store the performance results of the "
            "custom CUDA GEMM kernel.")
      .default_value(std::string());
  program.add_argument("-co", "--csv")
      .help("Path to the CSV file to store the performance results of the "
            "MatX GEMM kernel.")
      .default_value(std::string());

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  size_t m = program.get<int>("--m");
  size_t k = program.get<int>("--k");
  size_t n = program.get<int>("--n");

  const std::string cuda_csv = program.get<std::string>("--csv");
  const std::string precision = program.get<std::string>("--precision");

  if (precision == "fp64") {
    run_benchmark<double>(m, n, k, precision, cuda_csv, cuda_csv);
  } else if (precision == "fp32") {
    run_benchmark<float>(m, n, k, precision, cuda_csv, cuda_csv);
  } else {
    std::cerr << "Invalid precision. Please use one of the following: fp16, "
                 "fp32, fp64."
              << std::endl;
    std::exit(1);
  }

  return 0;
}

//template void run_benchmark<float>(size_t m, size_t n, size_t k,
//                                   const std::string &precision,
//                                   const std::string &cuda_csv,
//                                   const std::string &cublas_csv);