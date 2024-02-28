#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/execution_policy.h>


namespace thrust_vector {
// Functor with __host__ __device__ specifier
struct AddZeroFunctor {
    __host__ __device__
    float operator()(float val) const {
        return val + 0.0f;
    }
};

template <typename T>
struct IsEqualTo
{
  T value;

  __host__ __device__ bool operator()(T val) const
  {
    return val == value;
  }
};

}
