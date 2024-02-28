

SIZE_IN_MB=$1

rm -fr build

echo "*********************************************"
echo "Testing GCC with CUDA"
echo "*********************************************"

export CC=gcc
export CXX=g++

# Build and test all for GPU

cmake -S . -B build -DTHRUST_DEVICE_SYSTEM=CUDA
cmake --build build -j

# Run first tests

./build/thrust_bencher -n $SIZE_IN_MB -cpu 1 -gpu 1 -t_cpu 1 -t_gpu 1

rm -fr build

echo "*********************************************"
echo "Testing NVC with CUDA"
echo "*********************************************"

export CC=nvc
export CXX=nvc++

set_compiler nvc
Build and test all for GPU

cmake -S . -B build -DTHRUST_DEVICE_SYSTEM=CUDA
cmake --build build -j

./build/thrust_bencher -n $SIZE_IN_MB -cpu 1 -gpu 1 -t_cpu 1 -t_gpu 1 

