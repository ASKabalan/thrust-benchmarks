#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include <cuda_runtime.h>
#include <string>

template <typename T>
void launch_gemm_kernel_v07_vectorized(size_t m, size_t n, size_t k,
                                       T const *alpha, T const *A, size_t lda,
                                       T const *B, size_t ldb, T const *beta,
                                       T *C, size_t ldc, cudaStream_t stream);

template <typename T>
void launch_gemm_kernel_v06_vectorized(size_t m, size_t n, size_t k,
                                       T const *alpha, T const *A, size_t lda,
                                       T const *B, size_t ldb, T const *beta,
                                       T *C, size_t ldc, cudaStream_t stream);

template <typename T>
void profile_gemm(size_t m, size_t n, size_t k, T abs_tol, double rel_tol,
                  const std::string &precision, const std::string &cuda_csv,
                  const std::string &cublas_csv);



#endif