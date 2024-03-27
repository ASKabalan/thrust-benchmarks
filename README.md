# gemm benchmarks

python pallas/pallas_mm.py -m 4096 -n 4096 -k 4096 -p fp32 -o perf.csv
python jax/jax_mm.py -m 4096 -n 4096 -k 4096 -p fp32 -o perf.csv
./build/cuda/cuda_gemm -m 4096 -n 4096 -k 4096 -p fp32 -co perf.csv
./build/cutlass/cutlass_gemm -m 4096 -n 4096 -k 4096 -p fp32 -co perf.csv
./build/thrust/thrust_gemm -m 4096 -n 4096 -k 4096 -p fp32 -co perf.csv