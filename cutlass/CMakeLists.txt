add_executable(cutlass_gemm src/cutlass_gemm.cu)

target_include_directories(cutlass_gemm 
                            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
                            PRIVATE ${NVHPC_MATH_INCLUDE_DIR}
                            PRIVATE ${NVHPC_CUDA_INCLUDE_DIR}
)
target_link_libraries(cutlass_gemm PRIVATE NVHPC::CUDA NVHPC::CUBLAS)
set_target_properties(cutlass_gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})