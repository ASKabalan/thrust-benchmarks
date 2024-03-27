# Add CUDA executable

add_executable(cuda_gemm                
              ${CMAKE_CURRENT_LIST_DIR}/src/kernel_profiler.cu
              ${CMAKE_CURRENT_LIST_DIR}/src/cuda_gemm.cc  
              ${CMAKE_CURRENT_LIST_DIR}/src/cuda_gemm_kernel.cu
)


target_link_libraries(cuda_gemm PRIVATE NVHPC::CUDA NVHPC::CUBLAS)
target_include_directories(cuda_gemm PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
set_target_properties(cuda_gemm PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(cuda_gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})