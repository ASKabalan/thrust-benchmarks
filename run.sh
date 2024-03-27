#/!bin/bash

micromamba activate ax 
python jax/jax_mm.py -n 16384 -m 16384 -k 16384 -p fp32
./build/cuda/cuda_gemm -n 16384 -m 16384 -k 16384 -p fp32
./build/cutlass/cutlass_gemm -n 16384 -m 4096 -k 16384 -p fp32
./build/matx/matx_gemm -n 16384 -m 16384 -k 16384 -p fp32




