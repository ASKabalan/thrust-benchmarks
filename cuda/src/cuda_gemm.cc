#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "argparse.hpp"
#include "cuda_kernels.cuh"
#include <string.h>



int main(int argc, char *argv[]) {
  //print_device_info();

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
  program.add_argument("-co", "--cuda_csv")
      .help("Path to the CSV file to store the performance results of the "
            "custom CUDA GEMM kernel.")
      .default_value(std::string());
  program.add_argument("-cb", "--cublas_csv")
      .help("Path to the CSV file to store the performance results of the "
            "cuBLAS GEMM kernel.")
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

  const std::string cuda_csv = program.get<std::string>("--cuda_csv");
  const std::string cublas_csv = program.get<std::string>("--cublas_csv");
  const std::string precision = program.get<std::string>("--precision");

  // Define all the GEMM kernel launch functions to be profiled.
  if (precision == "fp32") {

    float const fp32_abs_tol{1.0e-3f};
    double const fp32_rel_tol{0.0e-4f};

    profile_gemm<float>(m, n, k, fp32_abs_tol, fp32_rel_tol, precision,
                        cuda_csv, cublas_csv);
  } else if (precision == "fp16") {

    __half const fp16_tensor_core_abs_tol{__float2half(5.0e-2f)};
    double const fp16_tensor_core_rel_tol{1.0e-1f};

    profile_gemm<__half>(m, n, k, fp16_tensor_core_abs_tol,
                         fp16_tensor_core_rel_tol, precision, cuda_csv,
                         cublas_csv);
  } else if (precision == "fp64") {
    double const fp64_abs_tol{1.0e-3};
    double const fp64_rel_tol{0.0e-4};

    profile_gemm<double>(m, n, k, fp64_abs_tol, fp64_rel_tol, precision,
                         cuda_csv, cublas_csv);
  } else {
    std::cerr << "Invalid precision. Please choose either fp32 or fp16."
              << std::endl;
    std::exit(1);
  }

  return 0;
}