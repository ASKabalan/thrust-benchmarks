# Add CUDA executable
add_executable(cublas_gemm ${CMAKE_CURRENT_LIST_DIR}/src/cublas_gemm.cu)
target_link_libraries(cublas_gemm PRIVATE NVHPC::CUDA NVHPC::CUBLAS)
target_include_directories(cublas_gemm PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
set_target_properties(cublas_gemm PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(cublas_gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})