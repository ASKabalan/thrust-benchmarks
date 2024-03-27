

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>

#include "checks.cuh"
#include "cuda_gemm_utils.cuh"
#include "cuda_kernels.cuh"
#include "perfostep.hpp"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUBLASS_ERROR(val) check_cublass((val), #val, __FILE__, __LINE__)
void check_cublass(cublasStatus_t err, const char* const func,
                   const char* const file, const int line)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Error at: " << file << ":" << line << std::endl;
        std::cerr << cublasGetStatusString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Determine CUDA data type from type.
template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, __half>::value,
                                  bool>::type = true>
constexpr cudaDataType_t cuda_data_type_trait() {
  if (std::is_same<T, float>::value) {
    return CUDA_R_32F;
  } else if (std::is_same<T, double>::value) {
    return CUDA_R_64F;
  } else if (std::is_same<T, __half>::value) {
    return CUDA_R_16F;
  } else {
    throw std::runtime_error("Unsupported data type.");
  }
}



template <typename T,
          typename std::enable_if<std::is_same<T, float>::value ||
                                      std::is_same<T, double>::value ||
                                      std::is_same<T, __half>::value,
                                  bool>::type = true>
void random_initialize_matrix(T *A, size_t m, size_t n, size_t lda,
                              unsigned int seed = 0U) {
  std::default_random_engine eng(seed);
  // The best way to verify is to use integer values.
  std::uniform_int_distribution<int> dis(0, 5);
  // std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  auto const rand = [&dis, &eng]() { return dis(eng); };
  for (size_t i{0U}; i < m; ++i) {
    for (size_t j{0U}; j < n; ++j) {
      A[i * lda + j] = static_cast<T>(rand());
    }
  }
}

template <typename T>
void launch_gemm_cublas(size_t m, size_t n, size_t k, T const *alpha,
                        T const *A, size_t lda, T const *B, size_t ldb,
                        T const *beta, T *C, size_t ldc,
                        cublasHandle_t handle) {
  // Non-TensorCore algorithm?
  constexpr cublasGemmAlgo_t algo{CUBLAS_GEMM_DEFAULT};
  constexpr cudaDataType_t data_type{cuda_data_type_trait<T>()};
  // All the matrix are in row-major order.
  // https://docs.nvidia.com/cuda/cublas/#cublasgemmex
  // A: m x k row-major -> A: k x m column-major non-transposed
  // B: k x n row-major -> B: n x k column-major non-transposed
  // C: m x n row-major -> C: n x m column-major non-transposed
  // Thus, without padding, the leading dimension of the matrix in row-major
  // order is the number of columns, i.e., k for A, n for B, and n for C.
  // Row-major order: C = AB + C
  // Column-major order: C = BA + C
  // The cuBLAS API requires the leading dimension of the matrix in
  // column-major order. This API call looks non-intuitive, but it is correct.
  CHECK_CUBLASS_ERROR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                   alpha, B, data_type, ldb, A, data_type, lda,
                                   beta, C, data_type, ldc, data_type, algo));
}

template <typename T>
void profile_gemm(size_t m, size_t n, size_t k, T abs_tol, double rel_tol,
                  const std::string &precision, const std::string &cuda_csv,
                  const std::string &cublas_csv) {
  T const alpha{static_cast<T>(1.0)};
  T const beta{static_cast<T>(1.0)};

  size_t lda{(k + 16U - 1U) / 16U * 16U};
  size_t ldb{(n + 16U - 1U) / 16U * 16U};
  size_t ldc{(n + 16U - 1U) / 16U * 16U};

  // Create CUDA stream.
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  // Allocate memory on host.
  T *A_host{nullptr};
  T *B_host{nullptr};
  T *C_host{nullptr};
  T *C_host_from_device{nullptr};
  CHECK_CUDA_ERROR(cudaMallocHost(&A_host, m * lda * sizeof(T)));
  CHECK_CUDA_ERROR(cudaMallocHost(&B_host, k * ldb * sizeof(T)));
  CHECK_CUDA_ERROR(cudaMallocHost(&C_host, m * ldc * sizeof(T)));
  CHECK_CUDA_ERROR(cudaMallocHost(&C_host_from_device, m * ldc * sizeof(T)));

  // Initialize matrix A and B.
  random_initialize_matrix(A_host, m, k, lda);
  random_initialize_matrix(B_host, k, n, ldb);
  random_initialize_matrix(C_host, m, n, ldc);

  // Allocate memory on device.
  T *A_device{nullptr};
  T *B_device{nullptr};
  T *C_device{nullptr};
  CHECK_CUDA_ERROR(cudaMalloc(&A_device, m * lda * sizeof(T)));
  CHECK_CUDA_ERROR(cudaMalloc(&B_device, k * ldb * sizeof(T)));
  CHECK_CUDA_ERROR(cudaMalloc(&C_device, m * ldc * sizeof(T)));

  // Copy matrix A and B from host to device.
  CHECK_CUDA_ERROR(cudaMemcpy(A_device, A_host, m * lda * sizeof(T),
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(B_device, B_host, k * ldb * sizeof(T),
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(C_device, C_host, m * ldc * sizeof(T),
                              cudaMemcpyHostToDevice));

  // Create cuBLAS handle.
  cublasHandle_t handle;
  CHECK_CUBLASS_ERROR(cublasCreate(&handle));
  CHECK_CUBLASS_ERROR(cublasSetStream(handle, stream));

  // Launch cuBLAS GEMM.
  Perfostep cublas_step;
  cublas_step.Start("cuBLAS GEMM");
  launch_gemm_cublas<T>(m, n, k, &alpha, A_device, lda, B_device, ldb, &beta,
                        C_device, ldc, handle);
  cudaDeviceSynchronize();
  double const latency_cublas = cublas_step.Stop();

  Perfostep cuda_step;
  cuda_step.Start("custom CUDA GEMM");
  launch_gemm_kernel_v06_vectorized(m, n, k, &alpha, A_device, lda, B_device,
                                    ldb, &beta, C_device, ldc, stream);
  cudaDeviceSynchronize();

  double const latency_cuda_gemm = cuda_step.Stop();

  // Release resources.
  CHECK_CUDA_ERROR(cudaFree(A_device));
  CHECK_CUDA_ERROR(cudaFree(B_device));
  CHECK_CUDA_ERROR(cudaFree(C_device));
  CHECK_CUDA_ERROR(cudaFreeHost(A_host));
  CHECK_CUDA_ERROR(cudaFreeHost(B_host));
  CHECK_CUDA_ERROR(cudaFreeHost(C_host));
  CHECK_CUDA_ERROR(cudaFreeHost(C_host_from_device));
  CHECK_CUBLASS_ERROR(cublasDestroy(handle));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

  static const std::string BANDWIDTH = "Effective Bandwidth (GB/s)";
  static const std::string TFLOPS = "Effective TFLOPS";
  // register the performance results.
  float const cuda_effective_bandwidth{
      compute_effective_bandwidth<T>(m, n, k, latency_cuda_gemm)};
  float const cuda_effective_tflops{
      compute_effective_tflops(m, n, k, latency_cuda_gemm)};
  float const cublas_effective_bandwidth{
      compute_effective_bandwidth<T>(m, n, k, latency_cublas)};
  float const cublas_effective_tflops{
      compute_effective_tflops(m, n, k, latency_cublas)};
  // add to perfostep
  const ColumnNames cuda_cols = {
      {"k", std::to_string(k)},
      {"m", std::to_string(m)},
      {"n", std::to_string(n)},
      {"Precision", precision},
      {BANDWIDTH, std::to_string(cuda_effective_bandwidth)},
      {TFLOPS, std::to_string(cuda_effective_tflops)}};
  const ColumnNames cublas_cols = {
      {"k", std::to_string(k)},
      {"m", std::to_string(m)},
      {"n", std::to_string(n)},
      {"Precision", precision},
      {BANDWIDTH, std::to_string(cublas_effective_bandwidth)},
      {TFLOPS, std::to_string(cublas_effective_tflops)}};

  cuda_step.UpdateCols(cuda_cols);
  cublas_step.UpdateCols(cublas_cols);

  // print the performance results.
  cuda_step.Report();
  cublas_step.Report();

  cuda_step.PrintToCSV(cuda_csv);
  cublas_step.PrintToCSV(cublas_csv);
}

// create  specializations
template
void profile_gemm<float>(size_t m, size_t n, size_t k, float abs_tol, double rel_tol,
                  const std::string &precision, const std::string &cuda_csv,
                  const std::string &cublas_csv);

template
void profile_gemm<double>(size_t m, size_t n, size_t k, double abs_tol, double rel_tol,
                  const std::string &precision, const std::string &cuda_csv,
                  const std::string &cublas_csv);

template
void profile_gemm<__half>(size_t m, size_t n, size_t k, __half abs_tol, double rel_tol,
                  const std::string &precision, const std::string &cuda_csv,
                  const std::string &cublas_csv);
