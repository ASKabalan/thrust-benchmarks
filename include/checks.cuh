#ifndef CUDA_MACRO_CHECKS_H
#define CUDA_MACRO_CHECKS_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <vector>

#ifdef __CUDACC__

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
inline void checkLast(const char *const file, const int line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define SYNCHRONIZE_AND_CHECK()                                                \
  cudaDeviceSynchronize();                                                     \
  CHECK_LAST_CUDA_ERROR();

#else

#define CHECK_CUDA_ERROR(val)
#define CHECK_LAST_CUDA_ERROR()
#define SYNCHRONIZE_AND_CHECK()

#endif 

#endif // CUDA_MACRO_CHECKS_H
